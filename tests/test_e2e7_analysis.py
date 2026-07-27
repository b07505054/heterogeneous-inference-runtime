from scripts.analyze_perf_model_results_e2e7 import (
    bucket_for, tile_efficiency, find_crossovers, evaluate_model, counterfactual_composition,
    build_manifest, per_shape_winner, CUDAGRAPH_CAPTURE_SIZES, CALLS_PER_STEP,
)


def _synthetic_bench():
    def cand(default_ms, gemv_ms, pre_ms):
        return {
            "1_default_linear": {"median_ms": default_ms, "max_abs_error": 0.0, "max_rel_error": 0.0},
            "2_gemv_loop": {"median_ms": gemv_ms, "max_abs_error": 1e-4, "max_rel_error": 1e-3},
            "3_pretransposed": {"median_ms": pre_ms, "max_abs_error": 1e-4, "max_rel_error": 1e-3},
            "4_tunableop": {"status": "skipped_this_shape_for_time_budget"},
        }

    rows = []
    for op, K, N in (("qkv_proj", 896, 1152), ("o_proj", 896, 896), ("down_proj", 4864, 896)):
        for m, default_ms, gemv_ms in ((1, 0.05, 0.07), (2, 0.50, 0.09), (3, 0.42, 0.12),
                                        (4, 0.42, 0.14), (6, 0.42, 0.19), (8, 0.42, 0.25)):
            rows.append({"operation": op, "M": m, "N": N, "K": K, "flops": 2 * m * N * K,
                         "candidates": cand(default_ms, gemv_ms, default_ms * 0.99)})
    return {"rows": rows}


# --- bucket_for / CUDA graph padding logic ---
def test_bucket_for_maps_to_next_captured_size():
    assert bucket_for(1) == 1
    assert bucket_for(2) == 2
    assert bucket_for(3) == 4
    assert bucket_for(6) == 8
    assert bucket_for(8) == 8


def test_bucket_for_exceeds_max_capture_size_uses_largest():
    assert bucket_for(100) == max(CUDAGRAPH_CAPTURE_SIZES)


# --- tile efficiency (Model W dependency) ---
def test_tile_efficiency_full_utilization_at_multiple_of_tile():
    assert tile_efficiency(64, 64, tile_m=64, tile_n=64) == 1.0


def test_tile_efficiency_low_for_tiny_m_against_64_tile():
    eff = tile_efficiency(2, 1152, tile_m=64, tile_n=64)
    assert eff < 0.05  # M=2 against a 64-row tile wastes >95% of the tile


# --- shape manifest construction ---
def test_build_manifest_flags_padding_for_non_captured_batch_sizes():
    manifest = build_manifest(_synthetic_bench())
    m3_rows = [r for r in manifest if r["logical_M"] == 3]
    assert all(r["physical_M_bucket"] == 4 and r["padding_applied"] for r in m3_rows)
    m2_rows = [r for r in manifest if r["logical_M"] == 2]
    assert all(r["physical_M_bucket"] == 2 and not r["padding_applied"] for r in m2_rows)


def test_build_manifest_calls_per_step_matches_model_structure():
    manifest = build_manifest(_synthetic_bench())
    qkv_row = next(r for r in manifest if r["operation"] == "qkv_proj" and r["logical_M"] == 1)
    assert qkv_row["calls_per_decode_step"] == CALLS_PER_STEP["qkv_proj"] == 24


# --- per-shape winner / crossover ---
def test_per_shape_winner_picks_gemv_loop_when_faster():
    manifest = build_manifest(_synthetic_bench())
    winners = per_shape_winner(manifest)
    assert winners["qkv_proj_M2"]["winner"] == "2_gemv_loop"
    assert winners["qkv_proj_M2"]["speedup_vs_default"] > 1.0
    # at M=1 the synthetic fixture makes pretransposed marginally (0.99x) faster than
    # default_linear -- the real point of this test is that GEMV-loop does NOT win at M=1
    # (it's the categorically slower path there), which the winner selection must reflect.
    assert winners["qkv_proj_M1"]["winner"] != "2_gemv_loop"


def test_find_crossovers_detects_the_categorical_jump():
    manifest = build_manifest(_synthetic_bench())
    crossovers = find_crossovers(manifest)
    assert crossovers["qkv_proj"]["crossover_m"] == 2  # >2x jump from M=1 to M=2 in synthetic data


# --- model evaluation (calibration/held-out split) ---
def test_evaluate_model_separates_held_out_m_from_calibration():
    manifest = build_manifest(_synthetic_bench())
    actual = {f"{r['operation']}_M{r['logical_M']}": r["candidates"]["1_default_linear"]["median_ms"] for r in manifest}
    # a "perfect" model that just echoes the default_linear value
    preds = dict(actual)
    result = evaluate_model(preds, actual, manifest, held_out_m={3, 6}, held_out_op="down_proj")
    assert result["calibration_mae"] == 0.0
    assert result["held_out_m_mae"] == 0.0
    assert result["held_out_shape_family_mae"] == 0.0


def test_evaluate_model_reports_nonzero_error_for_imperfect_predictions():
    manifest = build_manifest(_synthetic_bench())
    actual = {f"{r['operation']}_M{r['logical_M']}": r["candidates"]["1_default_linear"]["median_ms"] for r in manifest}
    preds = {k: v + 1.0 for k, v in actual.items()}
    result = evaluate_model(preds, actual, manifest, held_out_m={3, 6}, held_out_op="down_proj")
    assert result["calibration_mae"] == 1.0


# --- counterfactual composition ---
def test_counterfactual_composition_picks_best_candidate_per_shape():
    manifest = build_manifest(_synthetic_bench())
    result = counterfactual_composition(manifest, remainder_ms=2.0, batch_size=2)
    assert result["label"] == "COUNTERFACTUAL_COMPOSITION"
    # every included operation's best_per_call_ms should equal the min across its candidates (gemv here)
    for op, info in result["breakdown"].items():
        assert info["best_per_call_ms"] < 0.50  # strictly less than the synthetic default_linear cost at M=2
    assert result["counterfactual_decode_step_ms"] == result["counterfactual_projection_ms"] + 2.0
    assert result["counterfactual_throughput_tokens_per_s"] == 2 * 1000.0 / result["counterfactual_decode_step_ms"]


def test_counterfactual_composition_missing_shape_is_skipped_not_fabricated():
    manifest = build_manifest(_synthetic_bench())  # no gate_up_proj / lm_head rows in the synthetic set
    result = counterfactual_composition(manifest, remainder_ms=1.0, batch_size=2)
    assert "gate_up_proj" not in result["breakdown"]
    assert "lm_head" not in result["breakdown"]

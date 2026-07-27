import json

from scripts.generate_perf_model_workloads_e2e4 import exact_prompt_token_ids, FILLER
from scripts.analyze_perf_model_results_e2e4 import (
    per_round_labels, build_row, correlations, fit_and_evaluate_all, ranking_and_regret,
    CALIBRATION_PROMPT_LENGTHS, HELD_OUT_PROMPT_LENGTHS,
)
from scripts.analyze_perf_model_results import build_model_features
from perf_model.e2e3_test_helpers import synthetic_server_info
from perf_model import interference_scaling_models as ism


class _MockTokenizer:
    """Deterministic word-splitting tokenizer, avoids needing the real HF model for structure tests."""
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


# --- 1. prompt-length sweep generation (exact token counts) ---
def test_exact_prompt_token_ids_hits_exact_target_length():
    tok = _MockTokenizer()
    for target in (32, 64, 128, 256, 512):
        ids = exact_prompt_token_ids(tok, target)
        assert len(ids) == target


def test_exact_prompt_token_ids_uses_the_filler_text():
    assert "compiler" in FILLER


# --- 2. admitted-request multiplicity structure ---
def test_admission_pool_shape_matches_max_multiplicity():
    import scripts.generate_perf_model_workloads_e2e4 as gen
    tok = _MockTokenizer()
    pool = [gen.build_request(exact_prompt_token_ids(tok, 128), gen.ADMISSION_OUTPUT_TOKENS, f"ADMIT-128-{i}")
            for i in range(gen.MAX_MULTIPLICITY)]
    assert len(pool) == 2
    assert pool[0]["metadata"]["request_id"] != pool[1]["metadata"]["request_id"]
    assert all(len(r["prompt"]) == 128 for r in pool)


# --- 10. calibration/held-out split enforcement ---
def test_calibration_and_held_out_prompt_lengths_do_not_overlap():
    assert CALIBRATION_PROMPT_LENGTHS.isdisjoint(HELD_OUT_PROMPT_LENGTHS)
    assert CALIBRATION_PROMPT_LENGTHS == {64, 128, 512}
    assert HELD_OUT_PROMPT_LENGTHS == {256}


def _synthetic_row(prompt_length, mult, cand, interference):
    return {
        "prompt_length": prompt_length, "admitted_prompt_tokens": prompt_length * mult,
        "admitted_request_count": mult, "max_num_seqs": cand,
        "admitted_prefill_flops": prompt_length * mult * 1_000_000.0,
        "measured_new_request_prefill_ms": prompt_length * mult * 0.5,
        "interference_ms": interference, "peak_stall_ms": interference * 4 if interference else None,
        "total_stall_area_ms": interference * 50 if interference else None,
    }


def test_fit_and_evaluate_all_only_fits_on_calibration_rows():
    rows = [
        _synthetic_row(64, 1, 2, 80.0), _synthetic_row(128, 1, 2, 160.0), _synthetic_row(512, 1, 2, 640.0),
        _synthetic_row(256, 1, 2, 999999.0),  # held-out row with an extreme value that would badly skew a fit if leaked in
        _synthetic_row(64, 1, 1, 0.0),
    ]
    result = fit_and_evaluate_all(rows)
    assert result["n_calibration_rows"] == 3  # the 3 calibration-length, regime>1 rows
    assert result["n_held_out_rows"] == 1
    fitted_b = result["models"]["B_prompt_token_linear"]["fitted"]["params"]["C_token"]
    # true relationship in calibration data is exactly interference = 1.25 * admitted_prompt_tokens
    assert abs(fitted_b - 1.25) < 0.1  # would be wildly different if the held-out outlier leaked into the fit


# --- 12. candidate-ranking evaluation ---
def test_ranking_and_regret_reports_best_model_by_held_out_mae():
    rows = [_synthetic_row(64, 1, 2, 80.0), _synthetic_row(128, 1, 2, 160.0), _synthetic_row(512, 1, 2, 640.0),
            _synthetic_row(256, 1, 2, 320.0), _synthetic_row(256, 1, 8, 340.0)]
    model_eval = fit_and_evaluate_all(rows)
    ranking = ranking_and_regret(rows, model_eval)
    assert ranking["held_out_prompt_length"] == 256
    assert ranking["best_model_by_held_out_mae"] in ism.MODELS
    assert ranking["best_model_held_out_mae"] <= ranking["fixed_151ms_held_out_mae"] + 1e-6 or ranking["best_model_held_out_mae"] is not None


# --- 13. fail-closed / graceful handling of missing token timelines ---
def test_per_round_labels_skips_rounds_with_no_timeline_or_no_admissions():
    raw_no_timelines = {"anchor_timelines": [], "admission_pooled_rows": []}
    assert per_round_labels(raw_no_timelines) == []

    raw_no_admissions = {
        "anchor_timelines": [{"ok": True, "token_arrival_times": [0.0, 0.01, 0.02], "submit_time": -0.05,
                               "first_token_time": 0.0, "completion_time": 0.03, "output_tokens": 3}],
        "admission_pooled_rows": [],
    }
    assert per_round_labels(raw_no_admissions) == []


def test_build_row_handles_missing_server_info_and_metrics_gracefully():
    model = build_model_features()
    raw = {
        "prompt_length": 128, "multiplicity": 1, "max_num_seqs_requested": 2,
        "anchor_timelines": [], "admission_pooled_rows": [],
        "post_warmup_metrics_text": None, "final_metrics_text": None,
        "server_info_raw": None, "fixed_configuration": {"enable_chunked_prefill": True},
        "classification": "VALID", "reference_match": None, "oom_detected_in_log": False,
        "process_cleanup_status": "graceful_sigterm", "peak_gpu_memory_mib": 3000,
    }
    row = build_row(raw, model)
    assert row["interference_ms"] is None
    assert row["adherence"] is None
    assert row["measured_new_request_prefill_ms"] is None
    assert row["admitted_prefill_flops"] > 0  # FLOP estimate is analytical, doesn't need server_info


# --- 15. stable serialization of fitted model parameters ---
def test_fitted_model_json_round_trips_stably():
    fit_fn, predict_fn = ism.MODELS["B_prompt_token_linear"]
    rows = [_synthetic_row(64, 1, 2, 80.0), _synthetic_row(128, 1, 2, 160.0)]
    fitted = fit_fn(rows)
    blob = json.dumps(fitted.to_dict(), sort_keys=True)
    reparsed = json.loads(blob)
    reblob = json.dumps(reparsed, sort_keys=True)
    assert blob == reblob
    assert reparsed["params"]["C_token"] == fitted.params["C_token"]


# --- correlations helper (kernel-level connection) ---
def test_correlations_reports_pearson_r_for_each_target_feature_pair():
    rows = [_synthetic_row(64, 1, 2, 80.0), _synthetic_row(128, 1, 2, 160.0), _synthetic_row(512, 1, 2, 640.0),
            _synthetic_row(256, 2, 2, 700.0)]
    corr = correlations(rows)
    assert "interference_ms" in corr
    assert "admitted_prompt_tokens" in corr["interference_ms"]
    # perfectly proportional tokens vs interference in the first 3 rows should give a strong positive r
    assert corr["interference_ms"]["admitted_prompt_tokens"] > 0.7

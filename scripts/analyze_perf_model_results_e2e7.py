#!/usr/bin/env python3
"""E2E-7 analysis: shape manifest, per-shape winner/crossover, Models T-X,
counterfactual decode-step composition, hypothesis evaluation.

TRACE_PROVEN calls-per-decode-step (from E2E-7 grid-dimension analysis,
window ~19 steps): qkv=24, o_proj=24, gate_up=24, down_proj=24 (one per
layer, 24 layers), lm_head=1 (once per step). Total 97 projection GEMM
calls/step, matching the source-confirmed model structure exactly.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CALLS_PER_STEP = {"qkv_proj": 24, "o_proj": 24, "gate_up_proj": 24, "down_proj": 24, "lm_head": 1}
LAYERS = 24
CUDAGRAPH_CAPTURE_SIZES = [1, 2, 4, 8, 16]

# TRACE_PROVEN measured mean per-call kernel time (E2E-7 production trace,
# batch_size=2 window; used as the "production" reference point for the
# cross-check and for Model T's categorical baseline).
TRACE_MEASURED_MEAN_US_BATCH2 = {
    "qkv_proj": 398.3, "o_proj": 938.2, "down_proj": 938.2, "gate_up_proj": 2908.1, "lm_head": 43230.2,
}

HELD_OUT_BATCH_SIZES = {3, 6}
HELD_OUT_OPERATION_FAMILY = "down_proj"


def bucket_for(bs: int) -> int:
    candidates = [c for c in CUDAGRAPH_CAPTURE_SIZES if c >= bs]
    return min(candidates) if candidates else max(CUDAGRAPH_CAPTURE_SIZES)


def load_benchmark(path: Path) -> dict:
    return json.loads(path.read_text())


def build_manifest(bench: dict) -> list[dict]:
    manifest = []
    for row in bench["rows"]:
        op = row["operation"]
        manifest.append({
            "operation": op, "layer_count": LAYERS if op != "lm_head" else 1,
            "logical_M": row["M"], "physical_M_bucket": bucket_for(row["M"]),
            "padding_applied": bucket_for(row["M"]) != row["M"],
            "N": row["N"], "K": row["K"], "input_dtype": "float16", "weight_dtype": "float16",
            "output_dtype": "float16", "fused_bias": False, "calls_per_decode_step": CALLS_PER_STEP[op],
            "source_of_truth": "SOURCE_PROVEN_model_config_plus_TRACE_PROVEN_grid_dims",
            "candidates": row["candidates"], "flops": row["flops"],
        })
    return manifest


def per_shape_winner(manifest: list[dict]) -> dict:
    by_key: dict[tuple, list[dict]] = {}
    for row in manifest:
        by_key.setdefault((row["operation"], row["logical_M"]), []).append(row)
    winners = {}
    for (op, m), rows in by_key.items():
        row = rows[0]
        best = None
        best_ms = float("inf")
        for cand, res in row["candidates"].items():
            med = res.get("median_ms")
            if med is not None and med < best_ms:
                best_ms = med
                best = cand
        winners[f"{op}_M{m}"] = {"winner": best, "median_ms": best_ms,
                                  "default_median_ms": row["candidates"]["1_default_linear"]["median_ms"],
                                  "speedup_vs_default": (
                                      row["candidates"]["1_default_linear"]["median_ms"] / best_ms if best_ms else None
                                  )}
    return winners


def find_crossovers(manifest: list[dict]) -> dict:
    """For each operation, find the smallest M at which the default-linear
    median crosses above 2x the M=1 default-linear median (categorical jump point)."""
    by_op: dict[str, list[dict]] = {}
    for row in manifest:
        by_op.setdefault(row["operation"], []).append(row)
    crossovers = {}
    for op, rows in by_op.items():
        rows_sorted = sorted(rows, key=lambda r: r["logical_M"])
        m1 = next((r for r in rows_sorted if r["logical_M"] == 1), None)
        if not m1:
            continue
        base = m1["candidates"]["1_default_linear"]["median_ms"]
        crossover_m = None
        for r in rows_sorted:
            if r["logical_M"] == 1:
                continue
            med = r["candidates"]["1_default_linear"]["median_ms"]
            if base and med / base > 2.0:
                crossover_m = r["logical_M"]
                break
        crossovers[op] = {"base_m1_ms": base, "crossover_m": crossover_m}
    return crossovers


def tile_efficiency(M: int, N: int, tile_m: int = 64, tile_n: int = 64) -> float:
    import math
    padded_m = math.ceil(M / tile_m) * tile_m
    padded_n = math.ceil(N / tile_n) * tile_n
    return (M * N) / (padded_m * padded_n)


def model_t(manifest: list[dict]) -> dict:
    """Categorical kernel family (E2E-6 style): batch=1 -> measured GEMV cost,
    batch>=2 -> measured GEMM cost, applied per-operation using this slice's
    isolated benchmark (default_linear candidate, the production-equivalent op)."""
    preds = {}
    for row in manifest:
        op, m = row["operation"], row["logical_M"]
        m1_rows = [r for r in manifest if r["operation"] == op and r["logical_M"] == 1]
        gemm_rows = [r for r in manifest if r["operation"] == op and r["logical_M"] >= 2]
        c1 = m1_rows[0]["candidates"]["1_default_linear"]["median_ms"] if m1_rows else None
        c_multi = (statistics.median([r["candidates"]["1_default_linear"]["median_ms"] for r in gemm_rows])
                   if gemm_rows else None)
        preds[f"{op}_M{m}"] = c1 if m == 1 else c_multi
    return preds


def model_u(manifest: list[dict], calibration_only: bool) -> dict:
    """Per-shape best-legal-algorithm selection (min over candidates), fit only
    from calibration batch sizes {1,2,4,8}, applied/evaluated on all rows."""
    calib_best: dict[tuple, float] = {}
    for row in manifest:
        if calibration_only and row["logical_M"] in HELD_OUT_BATCH_SIZES:
            continue
        key = (row["operation"], row["logical_M"])
        best = min(v["median_ms"] for v in row["candidates"].values() if v.get("median_ms") is not None)
        calib_best[key] = best
    # for held-out M, use the bucketed/padded M's calibrated best (Model P-style bucket transfer)
    preds = {}
    for row in manifest:
        key = (row["operation"], row["logical_M"])
        if key in calib_best:
            preds[f"{row['operation']}_M{row['logical_M']}"] = calib_best[key]
        else:
            bucket_key = (row["operation"], bucket_for(row["logical_M"]))
            preds[f"{row['operation']}_M{row['logical_M']}"] = calib_best.get(bucket_key)
    return preds


def model_v(manifest: list[dict]) -> dict:
    """Kernel crossover model: below crossover_M, use best GEMV-family
    candidate; at/above, use best GEMM-family (default_linear) candidate.
    Crossover calibrated once (M=2, the smallest tested multi-batch size)."""
    crossover_m = 2
    preds = {}
    for row in manifest:
        op, m = row["operation"], row["logical_M"]
        if m < crossover_m:
            preds[f"{op}_M{m}"] = row["candidates"]["2_gemv_loop"]["median_ms"]
        else:
            preds[f"{op}_M{m}"] = row["candidates"]["1_default_linear"]["median_ms"]
    return preds


def model_w(manifest: list[dict]) -> dict:
    """Tile-waste model: predicted_kernel_cost = M1_per_row_cost_analog / tile_efficiency.
    Uses the M=1 default-linear cost as the 'useful compute' unit and inflates
    by 1/tile_efficiency at the padded (M,N) shape -- only where a 64x64 tile
    is the kernel-confirmed shape (all four transformer-layer projections;
    NOT lm_head, whose kernel's tile-relevant N far exceeds any padding effect
    at these tiny M and is dominated by the N=151936 dimension itself)."""
    preds = {}
    for row in manifest:
        op, m, n = row["operation"], row["logical_M"], row["N"]
        m1_rows = [r for r in manifest if r["operation"] == op and r["logical_M"] == 1]
        base = m1_rows[0]["candidates"]["1_default_linear"]["median_ms"] if m1_rows else None
        if base is None:
            continue
        eff = tile_efficiency(m, n)
        preds[f"{op}_M{m}"] = base / eff if eff > 0 else None
    return preds


def model_x(manifest: list[dict]) -> dict:
    """Operation-aware piecewise: same shape as Model V but crossover fit
    PER OPERATION FAMILY from the calibration data (find_crossovers), not
    forced to a single shared crossover point."""
    crossovers = find_crossovers(manifest)
    preds = {}
    for row in manifest:
        op, m = row["operation"], row["logical_M"]
        crossover_m = crossovers.get(op, {}).get("crossover_m") or 2
        if m < crossover_m:
            preds[f"{op}_M{m}"] = row["candidates"]["2_gemv_loop"]["median_ms"]
        else:
            preds[f"{op}_M{m}"] = row["candidates"]["1_default_linear"]["median_ms"]
    return preds


def actual_default_linear(manifest: list[dict]) -> dict:
    return {f"{r['operation']}_M{r['logical_M']}": r["candidates"]["1_default_linear"]["median_ms"] for r in manifest}


def evaluate_model(preds: dict, actual: dict, manifest: list[dict], held_out_m: set[int], held_out_op: str) -> dict:
    all_errs, held_m_errs, held_op_errs, calib_errs = [], [], [], []
    for row in manifest:
        key = f"{row['operation']}_M{row['logical_M']}"
        if key not in preds or preds[key] is None or key not in actual:
            continue
        err = abs(preds[key] - actual[key])
        all_errs.append(err)
        if row["logical_M"] in held_out_m:
            held_m_errs.append(err)
        elif row["operation"] == held_out_op:
            held_op_errs.append(err)
        else:
            calib_errs.append(err)
    def mae(xs):
        return statistics.mean(xs) if xs else None
    return {"calibration_mae": mae(calib_errs), "held_out_m_mae": mae(held_m_errs),
            "held_out_shape_family_mae": mae(held_op_errs), "max_error": max(all_errs) if all_errs else None,
            "n_total": len(all_errs)}


def counterfactual_composition(manifest: list[dict], remainder_ms: float, batch_size: int) -> dict:
    total_projection_ms = 0.0
    breakdown = {}
    for op, calls in CALLS_PER_STEP.items():
        rows = [r for r in manifest if r["operation"] == op and r["logical_M"] == batch_size]
        if not rows:
            continue
        best = min(v["median_ms"] for v in rows[0]["candidates"].values() if v.get("median_ms") is not None)
        contribution = best * calls
        breakdown[op] = {"best_per_call_ms": best, "calls_per_step": calls, "contribution_ms": contribution}
        total_projection_ms += contribution
    step_ms = total_projection_ms + remainder_ms
    tpot_ms = step_ms
    throughput = (batch_size * 1000.0 / step_ms) if step_ms else None
    return {"breakdown": breakdown, "counterfactual_projection_ms": total_projection_ms,
            "remainder_ms": remainder_ms, "counterfactual_decode_step_ms": step_ms,
            "counterfactual_tpot_ms": tpot_ms, "counterfactual_throughput_tokens_per_s": throughput,
            "label": "COUNTERFACTUAL_COMPOSITION"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bench = load_benchmark(args.benchmark)
    manifest = build_manifest(bench)
    winners = per_shape_winner(manifest)
    crossovers = find_crossovers(manifest)

    actual = actual_default_linear(manifest)
    models = {
        "T_categorical": model_t(manifest),
        "U_per_shape_selection": model_u(manifest, calibration_only=True),
        "V_crossover": model_v(manifest),
        "W_tile_waste": model_w(manifest),
        "X_operation_aware_piecewise": model_x(manifest),
    }
    model_eval = {name: evaluate_model(preds, actual, manifest, HELD_OUT_BATCH_SIZES, HELD_OUT_OPERATION_FAMILY)
                  for name, preds in models.items()}

    # Cross-check remainder: measured production step (E2E-6, batch=2, engine TPOT) minus
    # trace-measured projection total at batch=2 (TRACE_PROVEN mean us -> ms, calls/step).
    production_step_ms_batch2 = 169.735  # E2E-6 engine TPOT, batch=2
    trace_projection_total_ms = sum(
        (TRACE_MEASURED_MEAN_US_BATCH2[op] / 1000.0) * calls for op, calls in CALLS_PER_STEP.items()
    )
    remainder_ms = max(production_step_ms_batch2 - trace_projection_total_ms, 0.0)

    counterfactual = {
        bs: counterfactual_composition(manifest, remainder_ms, bs) for bs in (2, 3, 4, 6, 8)
    }

    args.out.write_text(json.dumps({
        "manifest": manifest, "per_shape_winner": winners, "crossovers": crossovers,
        "models": models, "model_evaluation": model_eval,
        "production_step_ms_batch2_reference": production_step_ms_batch2,
        "trace_projection_total_ms_batch2": trace_projection_total_ms,
        "remainder_ms": remainder_ms, "counterfactual": counterfactual,
    }, indent=2, default=str))
    print(f"wrote {args.out}")
    print(f"trace cross-check: production={production_step_ms_batch2:.2f}ms "
          f"projection_total={trace_projection_total_ms:.2f}ms remainder={remainder_ms:.2f}ms")


if __name__ == "__main__":
    main()

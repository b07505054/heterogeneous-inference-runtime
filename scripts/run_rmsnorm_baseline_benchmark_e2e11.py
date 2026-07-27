#!/usr/bin/env python3
"""E2E-11 Phase 3/4: RMSNorm completeness validation + optimization-effect
benchmark. For each of the 12 (tokens, hidden) shapes from the E2E-9 sweep,
measures:
  A. eager PyTorch decomposition (explicit, controlled function)
  B. torch.compile of that same function (compile/warmup excluded from timing)
  C. flashinfer.rmsnorm, if it actually runs on this GPU (real attempt, not assumed)
  D. custom CUDA RMSNorm at fixed block sizes 64/256/512
  E. custom CUDA RMSNorm via the full unified selector path (select_implementation)

Also runs full Phase-3 completeness: candidate generation -> legality ->
cost model -> selector -> static ExecutionPolicy -> JSON round trip ->
execute_rmsnorm -> correctness vs an independently-implemented reference,
and reports selector accuracy/regret against the real measured winner.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from perf_model.cost_model_registry import CostModelRegistry
from perf_model.execution_policy import ExecutionPolicy, build_static_policy
from perf_model.implementation_decision import select_implementation
from perf_model.operation_descriptor import (
    OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype, RMSNormDescriptor,
)
from perf_model.rmsnorm_cost_model_adapter import DEFAULT_EVIDENCE_PATH, RMSNormCostModel
from perf_model.runtime_dispatcher import execute_rmsnorm, torch_rmsnorm_eager, _load_rmsnorm_extension

WARMUP = 30
RUNS = 100
SHAPES = [(1, 768), (1, 1024), (1, 4096), (1, 8192),
          (16, 768), (16, 1024), (16, 4096), (16, 8192),
          (128, 768), (128, 1024), (128, 4096), (128, 8192)]


def timed_ms(fn, warmup=WARMUP, runs=RUNS) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    n = len(times)
    return {
        "median_ms": times[n // 2], "p95_ms": times[min(n - 1, int(0.95 * n))],
        "min_ms": times[0], "max_ms": times[-1],
        "stdev_ms": statistics.stdev(times) if n > 1 else 0.0, "n": n,
    }


def eager_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(orig_dtype)


def try_flashinfer(x: torch.Tensor, weight: torch.Tensor, eps: float):
    try:
        import flashinfer
        out = flashinfer.norm.rmsnorm(x, weight, eps)
        torch.cuda.synchronize()
        return out, None
    except Exception as exc:  # noqa: BLE001 -- report, don't hide
        return None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    assert torch.cuda.is_available()
    registry = CostModelRegistry()
    cost_model = RMSNormCostModel()
    registry.register(OperationFamily.RMS_NORM, cost_model)

    # one flashinfer feasibility probe, reported honestly either way
    probe_x = torch.randn(16, 4096, device="cuda", dtype=torch.float32)
    probe_w = torch.randn(4096, device="cuda", dtype=torch.float32)
    _, flashinfer_error = try_flashinfer(probe_x, probe_w, 1e-6)
    flashinfer_available = flashinfer_error is None
    print(f"flashinfer.rmsnorm feasibility on this GPU (cc={torch.cuda.get_device_capability()}): "
          f"{'AVAILABLE' if flashinfer_available else 'NOT AVAILABLE: ' + str(flashinfer_error)}")

    ext = _load_rmsnorm_extension()  # load once, outside all timed regions

    rows = []
    completeness_rows = []
    for tokens, hidden in SHAPES:
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32)
        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
        eps = 1e-6

        # ---- Phase 3: completeness path ----
        env = OperationEnvelope(operation_family=OperationFamily.RMS_NORM, operation_subtype=OperationSubtype.RMS_NORM_GENERIC,
                                 dtype="float32", device_type="cuda", target_arch="turing_sm75", phase="decode",
                                 logical_shape=(tokens, hidden))
        payload = RMSNormDescriptor(token_count=tokens, hidden_size=hidden, epsilon=eps, has_weight=True,
                                     input_contiguous=True, output_contiguous=True)
        op = OperationDescriptor(common=env, payload=payload)
        decision = select_implementation(op, registry, target={"dtype": "float32", "device_type": "cuda"})
        policy = build_static_policy(op, decision)
        policy2 = ExecutionPolicy.from_json(policy.to_json())
        roundtrip_ok = policy2.resolve_candidate_id() == policy.resolve_candidate_id()

        selected_out = execute_rmsnorm(x, weight, eps, decision)
        ref_out = torch_rmsnorm_eager(x, weight, eps)
        max_abs_err = (selected_out - ref_out).abs().max().item()
        max_rel_err = ((selected_out - ref_out).abs() / (ref_out.abs() + 1e-8)).max().item()
        correct = torch.allclose(selected_out, ref_out, rtol=1e-3, atol=1e-4)

        winner_bs, winner_ms = cost_model.measured_winner(tokens, hidden)
        selected_bs = decision.selected_candidate.parameters.get("block_size")
        completeness_rows.append({
            "tokens": tokens, "hidden": hidden, "selected_block_size": selected_bs, "measured_winner_block_size": winner_bs,
            "match": selected_bs == winner_bs, "correct": correct, "max_abs_error": max_abs_err, "max_rel_error": max_rel_err,
            "policy_roundtrip_ok": roundtrip_ok, "predicted_cost_source": decision.predicted_cost.source,
        })

        # ---- Phase 4: baseline benchmark ----
        eager_timing = timed_ms(lambda: eager_rmsnorm(x, weight, eps))

        compiled_fn = torch.compile(eager_rmsnorm)
        compiled_fn(x, weight, eps)  # compile + specialize, excluded from timing
        torch.cuda.synchronize()
        compiled_timing = timed_ms(lambda: compiled_fn(x, weight, eps))

        flashinfer_timing = None
        if flashinfer_available:
            import flashinfer
            flashinfer_timing = timed_ms(lambda: flashinfer.norm.rmsnorm(x, weight, eps))

        fixed_timings = {}
        for bs in (64, 256, 512):
            fixed_timings[bs] = timed_ms(lambda bs=bs: ext.fused_rmsnorm_forward(x, weight, eps, bs))

        selected_timing = timed_ms(lambda: execute_rmsnorm(x, weight, eps, decision))

        non_custom = {"eager": eager_timing["median_ms"], "torch_compile": compiled_timing["median_ms"]}
        if flashinfer_timing:
            non_custom["flashinfer"] = flashinfer_timing["median_ms"]
        fastest_non_custom_name = min(non_custom, key=non_custom.get)
        fastest_non_custom_ms = non_custom[fastest_non_custom_name]

        best_fixed_bs = min(fixed_timings, key=lambda bs: fixed_timings[bs]["median_ms"])
        best_fixed_ms = fixed_timings[best_fixed_bs]["median_ms"]

        custom_selected_speedup_vs_fastest_non_custom = fastest_non_custom_ms / selected_timing["median_ms"]
        selector_benefit_vs_best_fixed = best_fixed_ms / selected_timing["median_ms"]

        row = {
            "tokens": tokens, "hidden": hidden,
            "eager_median_ms": eager_timing["median_ms"], "eager_p95_ms": eager_timing["p95_ms"],
            "torch_compile_median_ms": compiled_timing["median_ms"], "torch_compile_p95_ms": compiled_timing["p95_ms"],
            "flashinfer_median_ms": flashinfer_timing["median_ms"] if flashinfer_timing else None,
            "custom_fixed_64_median_ms": fixed_timings[64]["median_ms"],
            "custom_fixed_256_median_ms": fixed_timings[256]["median_ms"],
            "custom_fixed_512_median_ms": fixed_timings[512]["median_ms"],
            "custom_selected_median_ms": selected_timing["median_ms"], "custom_selected_p95_ms": selected_timing["p95_ms"],
            "selected_block_size": selected_bs,
            "fastest_non_custom_name": fastest_non_custom_name, "fastest_non_custom_ms": fastest_non_custom_ms,
            "best_fixed_block_size": best_fixed_bs, "best_fixed_ms": best_fixed_ms,
            "custom_selected_speedup_vs_fastest_non_custom": custom_selected_speedup_vs_fastest_non_custom,
            "selector_benefit_vs_best_fixed_global_block": selector_benefit_vs_best_fixed,
        }
        rows.append(row)
        print(f"tokens={tokens:4d} hidden={hidden:5d} eager={eager_timing['median_ms']:.4f} "
              f"compile={compiled_timing['median_ms']:.4f} "
              f"flashinfer={flashinfer_timing['median_ms'] if flashinfer_timing else 'N/A'} "
              f"fixed64={fixed_timings[64]['median_ms']:.4f} fixed256={fixed_timings[256]['median_ms']:.4f} "
              f"fixed512={fixed_timings[512]['median_ms']:.4f} selected(bs={selected_bs})={selected_timing['median_ms']:.4f} "
              f"speedup_vs_fastest_noncustom={custom_selected_speedup_vs_fastest_non_custom:.3f}x "
              f"selector_vs_best_fixed={selector_benefit_vs_best_fixed:.3f}x")

    out = {
        "flashinfer_available": flashinfer_available, "flashinfer_error": flashinfer_error,
        "completeness_rows": completeness_rows, "benchmark_rows": rows,
        "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
    }
    out_path = Path(__file__).resolve().parents[1] / "results_e2e11_rmsnorm.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    n_match = sum(r["match"] for r in completeness_rows)
    n_correct = sum(r["correct"] for r in completeness_rows)
    print(f"\nSUMMARY: selector matches measured winner {n_match}/12, correct {n_correct}/12")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

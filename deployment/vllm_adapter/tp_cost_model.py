"""D5/D6: TP1-vs-TP2 throughput cost model -- offline calibration / oracle
reference utility.

D6 status (reclassified -- read this before using this module): this is
NOT part of the production TP-degree decision path. That decision is now
made by the real C++/MLIR compiler
(ml-graph-compiler-runtime/mlir_passes/lib/serving/
DistributedStrategyPlanningPass.cpp's distributed_profitability_contract_v1
selector), which reimplements this exact regression numerically (verified
bit-for-bit identical -- see
tools/generate_distributed_profitability_profile.py --check-against) and
consumes calibration coefficients from a versioned target-profile JSON,
never this Python module, at compile time.

This module's remaining, real roles are:
  1. Offline calibration fitting (`fit_linear_regression`,
     `TPCostModel.fit`) -- the actual source of the calibration
     coefficients embedded in
     ml-graph-compiler-runtime/configs/target_profiles/
     nvidia_rtx4090_d6_distributed_profitability.json, via
     tools/generate_distributed_profitability_profile.py.
  2. An independent oracle/reference implementation for cross-checking the
     compiler's C++ predictions against a separately-maintained Python
     implementation of the same math (see
     tests/test_distributed_d6_compiler_owned_tp_selection.py's
     production-path-independence tests).
  3. A regression-reference utility for anyone extending the calibration
     (e.g. adding a third model size) who wants to explore the fit in
     Python before regenerating the compiler-consumed profile.

No production script (scripts/run_d6_*.py, deployment/vllm_adapter/
distributed_materializer.py, or any real server-launch path) imports or
calls TPCostModel.decide() to choose a runtime TP degree. If you are
writing new code that launches a real vLLM server, the TP degree must
come from an ExecutionPlan's `distributed` block (produced by the real
compiler), never from calling this module directly.

Historical D5 note (mechanism unchanged from when this was the production
selector): predicts, from information available strictly *before*
execution (model identity, workload shape, GPU memory budget, and
constants regressed from already-collected calibration measurements),
which tensor-parallel degree a given workload should use. Never consults
held-out measurements. Never consults the actual outcome of the workload
being decided.

Two decision layers, in priority order:
  1. Feasibility (hard, analytical): if TP1 cannot legally hold the
     workload's weights + worst-case KV cache within the real GPU memory
     budget, TP2 is forced -- this is a closed-form memory calculation, not
     a regression, and is exactly the "TP2 capacity region" the spec asks
     us to search for.
  2. Performance (soft, regressed): if both TP degrees are feasible, a
     linear model -- fit ONLY on calibration-split throughput
     measurements -- predicts each degree's throughput for the workload's
     features; the compiler picks whichever is predicted higher.

Real, measured model-identity features (hidden_size, num_attention_heads,
num_kv_heads, num_layers) come directly from each model's real HF config
(deployment/execution_plan model_identity blocks / D3A-validated mapping).
weight_footprint_mb is the real measured/queried checkpoint size (see
distributed_materializer._estimate_model_footprint_mb and, for the 7B
entry, the real Hugging Face Hub blob-size total queried directly from
the Hub API on 2026-07-19 -- not a parameter-count guess).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

BYTES_PER_PARAM_FP16 = 2
KV_CACHE_HEADROOM_MB = 512.0  # matches distributed_materializer.KV_CACHE_HEADROOM_MB

# Real model-identity features. hidden_size/num_attention_heads/num_kv_heads/
# num_layers are the exact values used to build each model's real D2
# ExecutionPlan (results/runtime_paths/distributed_d2_qwen_pipeline/
# real_qwen*_execution_plan.json). weight_footprint_mb for the 0.5B model is
# the real value distributed_materializer._estimate_model_footprint_mb()
# measured against the actual cached checkpoint (1454.32 MB, matches every
# D5 0.5B sweep's own preflight artifact). weight_footprint_mb for the 7B
# model is the real total safetensors blob size returned by a live query
# against https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct?blobs=true
# (15,242,807,270 bytes = 14,539.06 MB) plus the same fixed headroom margin.
MODEL_IDENTITY_FEATURES: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "model_plan_id": "qwen2.5-0.5b", "hidden_size": 896, "num_attention_heads": 14,
        "num_kv_heads": 2, "num_layers": 24, "weight_footprint_mb": 1454.3235168457031,
        "max_position_embeddings": 32768,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "model_plan_id": "qwen2.5-7b", "hidden_size": 3584, "num_attention_heads": 28,
        "num_kv_heads": 4, "num_layers": 28,
        "weight_footprint_mb": 15242807270 / (1024 * 1024) + KV_CACHE_HEADROOM_MB,
        "max_position_embeddings": 32768,
    },
}

FEATURE_NAMES = (
    "per_gpu_weight_mb", "kv_cache_kb_per_token_per_gpu", "gpu_count",
    "input_length", "output_length", "concurrency",
)


def head_dim(model_features: dict[str, Any]) -> float:
    return model_features["hidden_size"] / model_features["num_attention_heads"]


def kv_cache_bytes_per_token_per_gpu(model_features: dict[str, Any], tp_degree: int) -> float:
    """2 (K & V) * num_layers * (num_kv_heads / tp_degree) * head_dim * fp16 bytes.

    vLLM's tensor-parallel attention shards KV heads across ranks, so each
    GPU only stores 1/tp_degree of the total KV cache for a given sequence
    -- this is the real mechanism (not an assumption) behind why splitting
    a model across GPUs can relieve per-GPU memory pressure even when
    weights are also (independently) sharded.
    """
    kv_heads_per_gpu = model_features["num_kv_heads"] / tp_degree
    return 2 * model_features["num_layers"] * kv_heads_per_gpu * head_dim(model_features) * BYTES_PER_PARAM_FP16


def per_gpu_weight_mb(model_features: dict[str, Any], tp_degree: int) -> float:
    """Real weight footprint divided across ranks. An approximation of
    vLLM's actual per-rank shard size (embedding/lm-head are not perfectly
    even-split), declared as such -- not claimed to be exact bytes."""
    return model_features["weight_footprint_mb"] / tp_degree


def is_feasible(
    model_features: dict[str, Any], tp_degree: int, *, gpu_total_mb: float, gpu_memory_utilization: float,
    max_model_len: int, max_num_seqs: int,
) -> tuple[bool, dict[str, float]]:
    """Closed-form worst-case memory check: can this TP degree legally hold
    its weight shard plus a full max_num_seqs x max_model_len KV cache
    within the real GPU memory budget? This is the hard capacity boundary
    -- computed, never regressed, and never treated as a performance
    preference."""
    budget_mb = gpu_total_mb * gpu_memory_utilization
    weight_mb = per_gpu_weight_mb(model_features, tp_degree)
    worst_case_kv_mb = (max_num_seqs * max_model_len * kv_cache_bytes_per_token_per_gpu(model_features, tp_degree)) / (1024 * 1024)
    required_mb = weight_mb + worst_case_kv_mb
    return required_mb <= budget_mb, {
        "budget_mb": budget_mb, "weight_mb": weight_mb, "worst_case_kv_mb": worst_case_kv_mb, "required_mb": required_mb,
    }


def build_feature_vector(
    model_features: dict[str, Any], tp_degree: int, *, input_length: int, output_length: int, concurrency: int,
) -> list[float]:
    return [
        per_gpu_weight_mb(model_features, tp_degree),
        kv_cache_bytes_per_token_per_gpu(model_features, tp_degree) / 1024.0,
        float(tp_degree),
        float(input_length), float(output_length), float(concurrency),
    ]


@dataclass
class FittedRegression:
    tp_degree: int
    coefficients: list[float]  # [intercept, *FEATURE_NAMES]
    n_samples: int
    r_squared: float

    def predict(self, feature_vector: list[float]) -> float:
        x = np.array([1.0] + list(feature_vector))
        return float(np.dot(x, self.coefficients))

    def to_dict(self) -> dict[str, Any]:
        return {"tp_degree": self.tp_degree, "coefficients": self.coefficients,
                "feature_names": ["intercept", *FEATURE_NAMES], "n_samples": self.n_samples,
                "r_squared": self.r_squared}


def fit_linear_regression(X: list[list[float]], y: list[float]) -> FittedRegression:
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    design = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
    coeffs, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    preds = design @ coeffs
    ss_res = float(np.sum((y_arr - preds) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return FittedRegression(tp_degree=-1, coefficients=coeffs.tolist(), n_samples=len(y), r_squared=r_squared)


@dataclass
class TPCostModel:
    """Frozen after fit(): must not be refit or adjusted after seeing
    held-out results."""

    throughput_models: dict[int, FittedRegression] = field(default_factory=dict)
    frozen: bool = False

    def fit(self, calibration_rows: list[dict[str, Any]]) -> None:
        if self.frozen:
            raise RuntimeError("cost model is frozen; cannot refit")
        for tp_degree in (1, 2):
            rows = [r for r in calibration_rows if r["tp_degree"] == tp_degree]
            X = [r["feature_vector"] for r in rows]
            y = [r["aggregate_throughput_tokens_per_s"] for r in rows]
            reg = fit_linear_regression(X, y)
            reg.tp_degree = tp_degree
            self.throughput_models[tp_degree] = reg
        self.frozen = True

    def predict_throughput(self, feature_vector: list[float], tp_degree: int) -> float:
        return self.throughput_models[tp_degree].predict(feature_vector)

    def decide(
        self, *, model_features: dict[str, Any], input_length: int, output_length: int, concurrency: int,
        gpu_total_mb: float, gpu_memory_utilization: float, max_model_len: int, max_num_seqs: int,
    ) -> dict[str, Any]:
        tp1_feasible, tp1_mem = is_feasible(model_features, 1, gpu_total_mb=gpu_total_mb,
                                             gpu_memory_utilization=gpu_memory_utilization,
                                             max_model_len=max_model_len, max_num_seqs=max_num_seqs)
        tp2_feasible, tp2_mem = is_feasible(model_features, 2, gpu_total_mb=gpu_total_mb,
                                             gpu_memory_utilization=gpu_memory_utilization,
                                             max_model_len=max_model_len, max_num_seqs=max_num_seqs)
        if not tp1_feasible and tp2_feasible:
            return {"decision": "tp2", "reason": "capacity_forced", "tp1_memory": tp1_mem, "tp2_memory": tp2_mem}
        if tp1_feasible and not tp2_feasible:
            return {"decision": "tp1", "reason": "capacity_forced", "tp1_memory": tp1_mem, "tp2_memory": tp2_mem}
        if not tp1_feasible and not tp2_feasible:
            return {"decision": "infeasible", "reason": "neither_tp_degree_fits", "tp1_memory": tp1_mem, "tp2_memory": tp2_mem}

        fv1 = build_feature_vector(model_features, 1, input_length=input_length, output_length=output_length, concurrency=concurrency)
        fv2 = build_feature_vector(model_features, 2, input_length=input_length, output_length=output_length, concurrency=concurrency)
        pred1 = self.predict_throughput(fv1, 1)
        pred2 = self.predict_throughput(fv2, 2)
        decision = "tp1" if pred1 >= pred2 else "tp2"
        return {
            "decision": decision, "reason": "performance_regression",
            "predicted_tp1_throughput": pred1, "predicted_tp2_throughput": pred2,
            "tp1_memory": tp1_mem, "tp2_memory": tp2_mem,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"frozen": self.frozen, "throughput_models": {k: v.to_dict() for k, v in self.throughput_models.items()}}

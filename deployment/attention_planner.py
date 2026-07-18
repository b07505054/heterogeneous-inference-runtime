"""Compiler planning boundary for ExecutionPlan-driven CPU attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import torch

from deployment.attention_runtime import make_attention_plan
from deployment.cpu_sharding import ShardingPlanError


SELECTOR_VERSION = "native_fused_attention_static_cost_selector_v3"
TARGET_PROFILE = "host_cpu_4_physical_8_logical_v1"
_ROOT = Path(__file__).resolve().parents[1]
_NATIVE_ARTIFACT = (
    _ROOT / "results/runtime_paths/native_fused_online_attention/"
    "libfused_online_attention.so")
_CANDIDATES = (
    ("dense_materialized", "torch_dense_materialized_v1", "serial", 1, 0, 0),
    ("dense_materialized", "torch_dense_materialized_v1", "split_head", 2, 0, 0),
    ("dense_materialized", "torch_dense_materialized_v1", "split_head", 4, 0, 0),
    ("dense_materialized", "torch_dense_materialized_v1", "split_query", 2, 0, 0),
    ("dense_materialized", "torch_dense_materialized_v1", "split_query", 4, 0, 0),
    ("dense_materialized", "torch_dense_materialized_v1", "split_query", 8, 0, 0),
    ("fused_tiled_online_softmax", "torch_tiled_online_softmax_exact_v1", "serial", 1, 1, 32),
    ("fused_tiled_online_softmax", "native_scalar", "serial", 1, 1, 32),
    ("fused_tiled_online_softmax", "native_scalar", "split_head", 2, 1, 32),
    ("fused_tiled_online_softmax", "native_scalar", "split_head", 4, 1, 32),
    ("fused_tiled_online_softmax", "native_avx2", "serial", 1, 1, 32),
    ("fused_tiled_online_softmax", "native_avx2", "split_head", 2, 1, 32),
    ("fused_tiled_online_softmax", "native_avx2", "split_head", 4, 1, 32),
)


@dataclass(frozen=True)
class AttentionWorkload:
    phase: str
    batch: int
    query_len: int
    context_len: int
    query_heads: int
    kv_heads: int
    head_dim: int
    dtype: str = "float32"
    causal: bool = True
    q_layout: str = "batch_head_sequence_dim"
    k_layout: str = "batch_head_sequence_dim"
    v_layout: str = "batch_head_sequence_dim"
    output_layout: str = "batch_head_sequence_dim"
    kv_cache_layout: str = "contiguous_layer_sequence_kv_head_dim"
    available_logical_workers: int = 8
    target_cpu_profile: str = TARGET_PROFILE
    selection_objective: str = "latency"
    temporary_memory_budget_bytes: int | None = None

    def validate(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise ShardingPlanError("attention workload phase must be prefill or decode")
        if min(self.batch, self.query_len, self.context_len, self.query_heads,
               self.kv_heads, self.head_dim, self.available_logical_workers) <= 0:
            raise ShardingPlanError("attention workload dimensions must be positive")
        if self.phase == "prefill" and self.query_len <= 1:
            raise ShardingPlanError("prefill workload requires query_len > 1")
        if self.phase == "decode" and self.query_len != 1:
            raise ShardingPlanError("decode workload requires query_len = 1")
        if self.query_heads % self.kv_heads:
            raise ShardingPlanError("query heads must be divisible by KV heads")
        if self.dtype != "float32" or self.head_dim != 64:
            raise ShardingPlanError("selector supports FP32 head_dim=64 only")
        layouts = (self.q_layout, self.k_layout, self.v_layout, self.output_layout)
        if any(x != "batch_head_sequence_dim" for x in layouts):
            raise ShardingPlanError("selector requires batch_head_sequence_dim layouts")
        if self.kv_cache_layout != "contiguous_layer_sequence_kv_head_dim":
            raise ShardingPlanError("selector requires contiguous model KV cache")
        if not self.causal:
            raise ShardingPlanError("selector supports causal attention only")
        if self.selection_objective not in {"latency", "memory_constrained"}:
            raise ShardingPlanError("unsupported attention selection objective")
        if self.selection_objective == "memory_constrained" and (
                not isinstance(self.temporary_memory_budget_bytes, int)
                or self.temporary_memory_budget_bytes <= 0):
            raise ShardingPlanError("memory-constrained selection requires a positive budget")

    @property
    def signature(self) -> str:
        return (
            f"b{self.batch}_q{self.query_len}_kv{self.context_len}_"
            f"qh{self.query_heads}_kvh{self.kv_heads}_d{self.head_dim}_fp32"
        )

    @classmethod
    def from_tensors(
        cls, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        *, available_logical_workers: int = 8,
        target_cpu_profile: str = TARGET_PROFILE,
    ) -> "AttentionWorkload":
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ShardingPlanError("actual attention Q/K/V must be rank 4")
        b, qh, qlen, dim = query.shape
        bk, kvh, context, kd = key.shape
        if bk != b or value.shape != key.shape or kd != dim:
            raise ShardingPlanError("actual Q/K/V shapes are incompatible")
        dtype = "float32" if query.dtype == key.dtype == value.dtype == torch.float32 else str(query.dtype)
        workload = cls(
            phase="decode" if qlen == 1 else "prefill", batch=b,
            query_len=qlen, context_len=context, query_heads=qh,
            kv_heads=kvh, head_dim=dim, dtype=dtype,
            available_logical_workers=available_logical_workers,
            target_cpu_profile=target_cpu_profile,
        )
        workload.validate()
        return workload


def _legality(
    workload: AttentionWorkload, algorithm: str, implementation: str,
    strategy: str, workers: int,
    query_tile: int, key_tile: int,
) -> tuple[bool, str]:
    if workers > workload.available_logical_workers:
        return False, "worker_count_exceeds_available_logical_workers"
    if strategy == "split_head" and workers > workload.query_heads:
        return False, "worker_count_exceeds_query_heads"
    if strategy == "split_query" and workload.phase == "decode":
        return False, "split_query_illegal_for_decode"
    if strategy == "split_query" and workers > workload.query_len:
        return False, "worker_count_exceeds_query_tokens"
    if algorithm == "fused_tiled_online_softmax":
        if query_tile <= 0 or key_tile <= 0:
            return False, "fused_tile_must_be_positive"
        if strategy == "split_query":
            return False, "fused_split_query_not_implemented"
        if workload.phase == "decode" and query_tile != 1:
            return False, "decode_requires_query_tile_1"
        if implementation in {"native_scalar", "native_avx2"}:
            if not _NATIVE_ARTIFACT.is_file():
                return False, "native_artifact_unavailable"
            if implementation == "native_avx2" and "avx2" not in torch.backends.cpu.get_cpu_capability().lower():
                return False, "target_isa_unavailable"
    return True, "legal"


def _score(
    workload: AttentionWorkload, algorithm: str, implementation: str,
    strategy: str, workers: int,
    query_tile: int, key_tile: int,
) -> dict[str, float]:
    compute_units = float(
        workload.batch * workload.query_heads * workload.query_len
        * workload.context_len * workload.head_dim
    )
    useful = 1 if strategy == "serial" else min(
        workers, workload.query_heads if strategy == "split_head" else workload.query_len
    )
    dispatch = 0.0 if strategy == "serial" else 750_000.0 * workers
    assembly = 0.0 if strategy == "serial" else (
        60_000.0 * workers if strategy == "split_head" else 80_000.0 * workers
    )
    memory = float(
        workload.batch * (workload.query_heads * workload.query_len
                          + 2 * workload.kv_heads * workload.context_len)
        * workload.head_dim * 4
    )
    score_temporary = (
        workload.batch * workload.query_heads * workload.query_len
        * workload.context_len * 4)
    probability_temporary = score_temporary
    tile_overhead = 0.0
    exponential_factor = 1.0
    if algorithm == "fused_tiled_online_softmax":
        score_temporary = 0
        probability_temporary = 0
        tiles = (
            math.ceil(workload.query_len / query_tile)
            * math.ceil(workload.context_len / key_tile))
        if implementation == "torch_tiled_online_softmax_exact_v1":
            tile_overhead = float(tiles * workload.query_heads * 45_000)
            exponential_factor = 1.12
        elif implementation == "native_scalar":
            tile_overhead = float(tiles * workload.query_heads * 1_100)
            exponential_factor = 1.45
        else:
            tile_overhead = float(tiles * workload.query_heads * 150)
            # Host calibration: AVX2 wins decode and tiny prefill, while its
            # scalar expf recurrence loses to dense matmul from q_len~32.
            exponential_factor = (
                0.45 if workload.phase == "decode" or workload.query_len < 32
                else 1.50)
    materialization_traffic = 2.0 * (score_temporary + probability_temporary)
    gqa_expansion = 0 if implementation.startswith("native_") else float(
        2 * workload.batch * workload.query_heads * workload.context_len
        * workload.head_dim * 4)
    if implementation.startswith("native_"):
        estimated_temporary = float((key_tile + workload.head_dim) * 4 * workers)
    elif algorithm == "dense_materialized":
        estimated_temporary = float(score_temporary + probability_temporary) + gqa_expansion
    else:
        estimated_temporary = float(
            workload.batch * workload.query_heads * min(query_tile, workload.query_len)
            * (2 + workload.head_dim + 2 * min(key_tile, workload.context_len)) * 4
        ) + gqa_expansion
    total = (
        compute_units * exponential_factor / useful + dispatch + assembly
        + memory * 0.02 + materialization_traffic * 0.35 + tile_overhead)
    return {
        "algorithm": algorithm,
        "implementation": implementation,
        "estimated_compute_units": compute_units,
        "estimated_qkv_output_bytes": memory,
        "estimated_score_temporary_bytes": float(score_temporary),
        "estimated_probability_temporary_bytes": float(probability_temporary),
        "estimated_materialization_traffic_bytes": materialization_traffic,
        "estimated_dispatch_cost": dispatch,
        "estimated_assembly_cost": assembly,
        "estimated_tile_overhead": tile_overhead,
        "estimated_exponential_factor": exponential_factor,
        "estimated_gqa_expansion_bytes": gqa_expansion,
        "estimated_total_temporary_bytes": estimated_temporary,
        "score": total,
    }


def select_attention_plan(workload: AttentionWorkload) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate, filter, score, and select one exact runtime candidate."""
    workload.validate()
    considered: list[dict[str, Any]] = []
    legal_plans: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for algorithm, implementation, strategy, workers, query_tile, key_tile in _CANDIDATES:
        legal, reason = _legality(
            workload, algorithm, implementation, strategy, workers, query_tile, key_tile)
        if algorithm == "dense_materialized":
            candidate_id = f"torch_cpu_attention_fp32_{strategy}_w{workers}_v1"
        elif implementation.startswith("native_"):
            candidate_id = (f"cpu_fused_online_{implementation}_fp32_q{query_tile}_"
                            f"k{key_tile}_{strategy}_w{workers}_v1")
        else:
            candidate_id = (f"torch_cpu_fused_online_fp32_q{query_tile}_k{key_tile}_"
                            f"{strategy}_w{workers}_v1")
        row: dict[str, Any] = {
            "candidate_id": candidate_id, "algorithm": algorithm,
            "implementation": implementation,
            "strategy": strategy, "worker_count": workers,
            "query_tile": query_tile, "key_tile": key_tile,
            "legal": legal, "legality_reason": reason,
        }
        if legal:
            row.update(_score(
                workload, algorithm, implementation, strategy, workers, query_tile, key_tile))
            if (workload.selection_objective == "memory_constrained"
                    and row["estimated_total_temporary_bytes"]
                    > workload.temporary_memory_budget_bytes):
                row["legal"] = False
                row["legality_reason"] = "temporary_memory_budget_exceeded"
                considered.append(row)
                continue
            native_fields = {}
            if implementation.startswith("native_"):
                from deployment.native_fused_attention import ABI_VERSION, SYMBOLS, sha256
                native_fields = {
                    "artifact_ref": str(_NATIVE_ARTIFACT),
                    "artifact_sha256": sha256(_NATIVE_ARTIFACT),
                    "abi_version": ABI_VERSION,
                    "native_symbol": SYMBOLS[implementation],
                }
            plan = make_attention_plan(
                phase=workload.phase, strategy=strategy, workers=workers,
                head_dim=workload.head_dim, query_heads=workload.query_heads,
                kv_heads=workload.kv_heads,
                provenance=SELECTOR_VERSION,
                algorithm=algorithm, query_tile=query_tile, key_tile=key_tile,
                implementation=implementation, **native_fields,
            )
            legal_plans.append((plan, row))
        considered.append(row)
    if not legal_plans:
        raise ShardingPlanError("compiler selector found no legal attention candidate")
    selected_plan, winner = min(
        legal_plans, key=lambda item: (item[1]["score"], item[1]["candidate_id"])
    )
    domain = {
        "batch": workload.batch,
        "query_length_min": workload.query_len,
        "query_length_max": workload.query_len,
        "context_length_min": workload.context_len,
        "context_length_max": workload.context_len,
    }
    selected_plan.update({
        "operator_kind": "attention",
        "selection_mode": "compiler_selected",
        "selector_version": SELECTOR_VERSION,
        "target_profile": workload.target_cpu_profile,
        "workload_signature": workload.signature,
        "workload_domain": domain,
        "query_length_domain": [workload.query_len, workload.query_len],
        "context_length_domain": [workload.context_len, workload.context_len],
        "fallback_status": False,
    })
    trace = {
        "selector_version": SELECTOR_VERSION,
        "selection_mode": "compiler_selected",
        "selection_provenance": SELECTOR_VERSION,
        "workload": asdict(workload),
        "workload_signature": workload.signature,
        "generated_candidate_count": len(_CANDIDATES),
        "legal_candidate_count": len(legal_plans),
        "rejected_candidates": [x for x in considered if not x["legal"]],
        "considered_candidates": considered,
        "selected_candidate_id": selected_plan["native_kernel_id"],
        "selected_score": winner["score"],
        "fallback_status": False,
        "tie_breaking": "minimum_score_then_lexicographic_candidate_id",
    }
    selected_plan["selection_trace"] = trace
    return selected_plan, trace


def widen_context_domain(plan: dict[str, Any], minimum: int, maximum: int) -> dict[str, Any]:
    if minimum <= 0 or maximum < minimum:
        raise ShardingPlanError("invalid context domain")
    widened = dict(plan)
    widened["workload_domain"] = dict(plan["workload_domain"])
    widened["workload_domain"]["context_length_min"] = minimum
    widened["workload_domain"]["context_length_max"] = maximum
    widened["context_length_domain"] = [minimum, maximum]
    return widened


def force_test_attention_plan(
    workload: AttentionWorkload, *, algorithm: str, strategy: str,
    workers: int, query_tile: int, key_tile: int,
    implementation: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Explicit diagnostic override; never reported as compiler-selected."""
    workload.validate()
    implementation = implementation or (
        "torch_dense_materialized_v1" if algorithm == "dense_materialized"
        else "torch_tiled_online_softmax_exact_v1")
    legal, reason = _legality(
        workload, algorithm, implementation, strategy, workers, query_tile, key_tile)
    if not legal:
        raise ShardingPlanError(f"forced test attention candidate is illegal: {reason}")
    native_fields = {}
    if implementation.startswith("native_"):
        from deployment.native_fused_attention import ABI_VERSION, SYMBOLS, sha256
        native_fields = {"artifact_ref": str(_NATIVE_ARTIFACT),
            "artifact_sha256": sha256(_NATIVE_ARTIFACT), "abi_version": ABI_VERSION,
            "native_symbol": SYMBOLS[implementation]}
    plan = make_attention_plan(
        phase=workload.phase, strategy=strategy, workers=workers,
        head_dim=workload.head_dim, query_heads=workload.query_heads,
        kv_heads=workload.kv_heads, provenance="forced_test_override",
        algorithm=algorithm, query_tile=query_tile, key_tile=key_tile,
        implementation=implementation, **native_fields)
    domain = {
        "batch": workload.batch,
        "query_length_min": workload.query_len,
        "query_length_max": workload.query_len,
        "context_length_min": workload.context_len,
        "context_length_max": workload.context_len,
    }
    trace = {
        "selector_version": SELECTOR_VERSION,
        "selection_mode": "forced_test_override",
        "selection_provenance": "forced_test_override",
        "workload": asdict(workload),
        "workload_signature": workload.signature,
        "generated_candidate_count": len(_CANDIDATES),
        "legal_candidate_count": 1,
        "rejected_candidates": [],
        "considered_candidates": [{
            "candidate_id": plan["native_kernel_id"], "algorithm": algorithm,
            "implementation": implementation,
            "strategy": strategy, "worker_count": workers,
            "query_tile": query_tile, "key_tile": key_tile,
            "legal": True, "legality_reason": "forced_legal_test_candidate"}],
        "selected_candidate_id": plan["native_kernel_id"],
        "selected_score": None,
        "fallback_status": False,
        "tie_breaking": "not_applicable_forced_test_override",
    }
    plan.update({
        "operator_kind": "attention",
        "selection_mode": "forced_test_override",
        "selector_version": SELECTOR_VERSION,
        "target_profile": workload.target_cpu_profile,
        "workload_signature": workload.signature,
        "workload_domain": domain,
        "query_length_domain": [workload.query_len, workload.query_len],
        "context_length_domain": [workload.context_len, workload.context_len],
        "fallback_status": False,
        "selection_trace": trace,
    })
    return plan, trace


def emit_execution_plan(
    *, plan_id: str, model_id: str, prompt_tokens: int, generated_tokens: int,
    prefill: dict[str, Any], decode: dict[str, Any],
) -> dict[str, Any]:
    table = {
        "decision_kind": "cpu_attention_plan_table_v1",
        "operator_kind": "attention",
        "operator_id": "qwen.layers.*.self_attn",
        "plan_kind": "phase_specific_exact_prefill_decode_context_range",
        "selector_version": SELECTOR_VERSION,
        "selection_mode": prefill["selection_mode"],
        "target_profile": TARGET_PROFILE,
        "phase_decisions": {"prefill": prefill, "decode": decode},
        "fallback": {"policy": "hard_failure", "count": 0},
        "runtime_no_redecision": True,
    }
    return {
        "schema": "execution_plan",
        "schema_version": "2.0.0",
        "plan_id": plan_id,
        "provenance": {
            "compiler_tool": "attention_planner.py",
            "model_spec_ref": model_id,
            "capability_bundle": {
                "hardware_profile_ref": TARGET_PROFILE,
                "backend_profile_refs": ["transformers_attention_registry"],
                "kernel_profile_refs": ["torch_cpu_attention_fp32_v1"],
                "workload_ref": f"prompt_tokens={prompt_tokens},generated_tokens={generated_tokens}",
            },
            "truth_boundary": "compiler_selected_real_model_attention_not_vllm_serving",
        },
        "model_identity": {
            "model_id": model_id, "dtype": "float32",
            "architecture": "Qwen2ForCausalLM",
        },
        "global_decisions": {
            "quantization": {"strategy": "none", "dtype": "float32"},
            "memory": {"kv_cache_layout": "contiguous"},
            "serving": {"parallelism_kind": "none", "parallelism_degree": 1},
            "attention_execution": table,
        },
        "function_plans": [],
    }

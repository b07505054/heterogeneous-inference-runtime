"""Loader and validator for compiler ExecutionPlan v2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deployment.execution_plan.schema import KNOWN_COLLECTIVE_KINDS, ExecutionPlan


class ExecutionPlanError(ValueError):
    """Raised when an ExecutionPlan v2 artifact is malformed or contaminated."""


_MEASURED_FIELDS = {
    "measured_latency_ms",
    "actual_latency_ms",
    "measured_memory_mb",
    "measured_speedup",
    "speedup",
    "performance_claim",
    "runtime_result",
    "metrics",
}


def load_execution_plan(path: str | Path) -> ExecutionPlan:
    plan_path = Path(path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExecutionPlanError(f"execution plan not found: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise ExecutionPlanError(f"invalid execution plan JSON: {exc}") from exc
    return parse_execution_plan(payload)


def parse_execution_plan(payload: dict[str, Any]) -> ExecutionPlan:
    validate_execution_plan(payload)
    return ExecutionPlan.from_dict(payload)


def validate_execution_plan(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        raise ExecutionPlanError("ExecutionPlan v2 must be a JSON object")
    if payload.get("schema") != "execution_plan":
        errors.append("schema must be execution_plan")
    if payload.get("schema_version") != "2.0.0":
        errors.append("schema_version must be 2.0.0")
    for key in ("plan_id", "provenance", "model_identity", "global_decisions", "function_plans"):
        if key not in payload:
            errors.append(f"missing required field: {key}")
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict) or "capability_bundle" not in provenance:
        errors.append("missing required field: provenance.capability_bundle")
    if not isinstance(payload.get("function_plans", []), list):
        errors.append("function_plans must be a list")
    if _contains_measured_field(payload):
        errors.append("compiler execution plan must not contain measured runtime fields")
    errors.extend(_validate_exact_rmsnorm_decisions(payload))
    errors.extend(_validate_aarch64_native_decisions(payload))
    sharding = (payload.get("global_decisions") or {}).get("cpu_sharding")
    if sharding is not None:
        try:
            from deployment.cpu_sharding import validate_sharding
            validate_sharding(sharding)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid global_decisions.cpu_sharding: {exc}")
    attention = (payload.get("global_decisions") or {}).get("attention_execution")
    if attention is not None:
        try:
            from deployment.attention_runtime import validate_attention_execution
            validate_attention_execution(attention)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid global_decisions.attention_execution: {exc}")
    errors.extend(_validate_paged_kv_execution_contracts(payload))
    if "distributed" in payload:
        errors.extend(_validate_distributed_plan(payload.get("distributed")))
    if errors:
        raise ExecutionPlanError("; ".join(errors))
    return []


def _validate_paged_kv_execution_contracts(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for function in payload.get("function_plans", []):
        if not isinstance(function, dict):
            continue
        for op in function.get("per_op_decisions", []):
            if not isinstance(op, dict) or "paged_kv_execution" not in op:
                continue
            contract = op.get("paged_kv_execution")
            prefix = (
                f"function_plans.{function.get('function_name', '<unknown>')}"
                f".per_op_decisions.{op.get('op_name', '<unknown>')}"
                ".paged_kv_execution"
            )
            if not isinstance(contract, dict):
                errors.append(f"{prefix} must be an object")
                continue
            errors.extend(
                f"{prefix}: {error}"
                for error in _validate_one_paged_kv_execution(contract)
            )
    return errors


def _validate_one_paged_kv_execution(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "kv_execution_unit",
        "kv_candidate_id",
        "kv_layout_kind",
        "dtype",
        "batch",
        "num_kv_heads",
        "head_dim",
        "page_tokens",
        "num_physical_pages",
        "maximum_logical_tokens",
        "maximum_logical_blocks",
        "block_table_length",
        "block_table_element_type",
        "invalid_page_sentinel",
        "bytes_per_token",
        "bytes_per_k_page",
        "bytes_per_v_page",
        "bytes_per_combined_page",
        "total_pool_bytes",
        "alignment_bytes",
        "k_page_strides",
        "v_page_strides",
        "pool_artifact_ref",
        "pool_artifact_sha256",
        "pool_artifact_version",
        "pool_create_entry_point",
        "prefill_write_entry_point",
        "append_entry_point",
        "view_binding",
        "reset_entry_point",
        "release_entry_point",
        "paged_attention_kernel_id",
        "paged_attention_entry_point",
        "runtime_no_layout_redecision",
        "runtime_no_kernel_redecision",
    )
    for key in required:
        if key not in contract:
            errors.append(f"missing required field: {key}")

    def int_field(key: str) -> int | None:
        value = contract.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{key} must be an integer")
            return None
        return value

    page_tokens = int_field("page_tokens")
    num_pages = int_field("num_physical_pages")
    max_tokens = int_field("maximum_logical_tokens")
    max_blocks = int_field("maximum_logical_blocks")
    table_len = int_field("block_table_length")
    heads = int_field("num_kv_heads")
    head_dim = int_field("head_dim")
    batch = int_field("batch")
    bytes_per_token = int_field("bytes_per_token")
    bytes_per_k_page = int_field("bytes_per_k_page")
    bytes_per_v_page = int_field("bytes_per_v_page")
    bytes_per_combined_page = int_field("bytes_per_combined_page")
    total_pool_bytes = int_field("total_pool_bytes")
    alignment = int_field("alignment_bytes")
    sentinel = int_field("invalid_page_sentinel")

    if contract.get("kv_execution_unit") != "portable_cpu_paged_kv":
        errors.append("kv_execution_unit must be portable_cpu_paged_kv")
    if contract.get("kv_layout_kind") != "paged_phd_contiguous":
        errors.append("unsupported kv_layout_kind")
    if contract.get("dtype") != "fp32":
        errors.append("unsupported dtype")
    if batch != 1:
        errors.append("batch must be 1 for current native paged-KV ABI")
    if page_tokens is not None and page_tokens <= 0:
        errors.append("page_tokens must be > 0")
    if num_pages is not None and num_pages <= 0:
        errors.append("num_physical_pages must be > 0")
    if max_tokens is not None and max_tokens <= 0:
        errors.append("maximum_logical_tokens must be > 0")
    if heads is not None and heads <= 0:
        errors.append("num_kv_heads must be > 0")
    if head_dim is not None and head_dim <= 0:
        errors.append("head_dim must be > 0")
    if alignment is not None and alignment <= 0:
        errors.append("alignment_bytes must be > 0")
    if contract.get("block_table_element_type") != "int32":
        errors.append("block_table_element_type must be int32")
    if sentinel is not None and not (-(2**31) <= sentinel <= 2**31 - 1):
        errors.append("invalid_page_sentinel must fit int32")

    if (
        page_tokens is not None
        and num_pages is not None
        and max_tokens is not None
        and max_tokens > page_tokens * num_pages
    ):
        errors.append("maximum_logical_tokens exceeds physical page capacity")
    if page_tokens and max_tokens and max_blocks is not None:
        expected_blocks = (max_tokens + page_tokens - 1) // page_tokens
        if max_blocks != expected_blocks:
            errors.append("maximum_logical_blocks formula mismatch")
        if table_len is not None and table_len < expected_blocks:
            errors.append("block_table_length smaller than logical blocks")

    if heads and head_dim and bytes_per_token is not None:
        expected_bpt = 2 * heads * head_dim * 4
        if bytes_per_token != expected_bpt:
            errors.append("bytes_per_token formula mismatch")
    if heads and page_tokens and head_dim and bytes_per_k_page is not None:
        expected_page = heads * page_tokens * head_dim * 4
        if bytes_per_k_page != expected_page:
            errors.append("bytes_per_k_page formula mismatch")
        if bytes_per_v_page != expected_page:
            errors.append("bytes_per_v_page formula mismatch")
        if bytes_per_combined_page != 2 * expected_page:
            errors.append("bytes_per_combined_page formula mismatch")
        if num_pages and total_pool_bytes != num_pages * 2 * expected_page:
            errors.append("total_pool_bytes formula mismatch")
        expected_strides = [heads * page_tokens * head_dim, page_tokens * head_dim, head_dim, 1]
        if contract.get("k_page_strides") != expected_strides:
            errors.append("k_page_strides formula mismatch")
        if contract.get("v_page_strides") != expected_strides:
            errors.append("v_page_strides formula mismatch")

    if contract.get("runtime_no_layout_redecision") is not True:
        errors.append("runtime_no_layout_redecision must be true")
    if contract.get("runtime_no_kernel_redecision") is not True:
        errors.append("runtime_no_kernel_redecision must be true")
    if contract.get("contiguous_fallback_identity") not in (
        "",
        "cpu_contiguous_kv_fp32_v1",
    ):
        errors.append("unsupported fallback identity")
    if "num_query_heads" in contract and contract.get("num_query_heads") != heads:
        errors.append("num_query_heads must equal num_kv_heads for current native kernel")
    if "workspace_bytes" in contract:
        workspace = int_field("workspace_bytes")
        if workspace is not None and workspace < 0:
            errors.append("workspace_bytes must be non-negative")

    expected_entries = {
        "pool_create_entry_point": "hir_paged_kv_initialize",
        "prefill_write_entry_point": "hir_paged_kv_prefill_write",
        "append_entry_point": "hir_paged_kv_append",
        "reset_entry_point": "hir_paged_kv_reset",
        "release_entry_point": "runtime_owned_pool_release",
        "view_binding": "direct_int32_block_table_translation",
    }
    for key, expected in expected_entries.items():
        if contract.get(key) != expected:
            errors.append(f"{key} must be {expected}")
    variants = {
        "cpu_paged_kv_fp32_v1": (
            "cpu_attention_decode_paged_kv_fp32",
            "hir_cpu_attention_decode_paged_kv_fp32",
        ),
        "cpu_paged_kv_fp32_token_major_v1": (
            "cpu_attention_decode_paged_kv_fp32",
            "hir_cpu_attention_decode_paged_kv_fp32",
        ),
        "cpu_paged_kv_fp32_page_major_v1": (
            "cpu_attention_decode_paged_kv_page_major_fp32",
            "hir_cpu_attention_decode_paged_kv_page_major_fp32",
        ),
    }
    expected_kernel, expected_entry = variants.get(
        contract.get("kv_candidate_id"),
        ("", ""),
    )
    if not expected_kernel:
        errors.append("unsupported kv_candidate_id")
    if contract.get("paged_attention_kernel_id") != expected_kernel:
        errors.append("paged_attention_kernel_id does not match candidate")
    if contract.get("paged_attention_entry_point") != expected_entry:
        errors.append("paged_attention_entry_point does not match candidate")
    if contract.get("pool_artifact_version") != "hir.paged_kv.v1":
        errors.append("pool_artifact_version must be hir.paged_kv.v1")
    for key in (
        "pool_artifact_ref",
        "pool_artifact_sha256",
        "pool_create_entry_point",
        "prefill_write_entry_point",
        "append_entry_point",
        "paged_attention_entry_point",
    ):
        if not contract.get(key):
            errors.append(f"{key} must be non-empty")
    return errors


def _validate_distributed_plan(distributed: Any) -> list[str]:
    """D1: fail-closed structural validation of ExecutionPlan.distributed.

    Mirrors ml-graph-compiler-runtime's DistributedPlanning::validateDistributedPlan
    legality rules. Any malformed TP=2 plan is rejected here -- the runtime
    never silently downgrades to TP=1 and never silently ignores an unknown
    collective kind.
    """
    errors: list[str] = []
    if not isinstance(distributed, dict):
        return ["distributed must be an object when present"]

    world_size = distributed.get("world_size")
    tp_size = distributed.get("tensor_parallel_size")
    pp_size = distributed.get("pipeline_parallel_size")
    if not isinstance(world_size, int) or world_size < 1:
        errors.append("distributed.world_size must be an integer >= 1")
    if not isinstance(tp_size, int) or tp_size < 1:
        errors.append("distributed.tensor_parallel_size must be an integer >= 1")
    if pp_size != 1:
        errors.append("distributed.pipeline_parallel_size must be == 1 for D1")
    if isinstance(world_size, int) and isinstance(tp_size, int) and world_size != tp_size:
        errors.append("distributed.world_size must equal tensor_parallel_size for D1")

    ranks = distributed.get("ranks", [])
    if not isinstance(ranks, list):
        errors.append("distributed.ranks must be a list")
        ranks = []
    rank_ids = [r.get("rank_id") for r in ranks if isinstance(r, dict)]
    declared_ranks = set(rank_ids)
    if len(declared_ranks) != len(rank_ids):
        errors.append("distributed.ranks contains duplicate rank_id")
    if isinstance(world_size, int) and (
        len(ranks) != world_size
        or declared_ranks != set(range(world_size))
    ):
        errors.append(
            "distributed.ranks must be exactly the contiguous set "
            "0..world_size-1"
        )

    collectives = distributed.get("collectives", [])
    if not isinstance(collectives, list):
        errors.append("distributed.collectives must be a list")
        collectives = []
    seen_sequence_ids: list[int] = []
    for step in collectives:
        if not isinstance(step, dict):
            errors.append("distributed.collectives entries must be objects")
            continue
        kind = step.get("kind")
        if kind not in KNOWN_COLLECTIVE_KINDS:
            errors.append(
                f"distributed collective '{step.get('collective_id')}' has "
                f"unknown or unsupported kind: {kind!r} "
                "(runtime must fail closed, never silently ignore it)"
            )
        seq = step.get("sequence_id")
        if not isinstance(seq, int):
            errors.append(
                f"distributed collective '{step.get('collective_id')}' is "
                "missing an integer sequence_id"
            )
        else:
            seen_sequence_ids.append(seq)
        participants = step.get("participants", [])
        if not isinstance(participants, list) or not participants:
            errors.append(
                f"distributed collective '{step.get('collective_id')}' must "
                "declare a non-empty participants list"
            )
        else:
            if len(set(participants)) != len(participants):
                errors.append(
                    f"distributed collective '{step.get('collective_id')}' "
                    "has duplicate participants"
                )
            if declared_ranks and not set(participants) <= declared_ranks:
                errors.append(
                    f"distributed collective '{step.get('collective_id')}' "
                    "references a rank not declared in distributed.ranks"
                )
    if seen_sequence_ids:
        if len(set(seen_sequence_ids)) != len(seen_sequence_ids):
            errors.append("distributed.collectives has duplicate sequence_id")
        if sorted(seen_sequence_ids) != list(range(len(seen_sequence_ids))):
            errors.append(
                "distributed.collectives sequence_ids must be exactly "
                "0..N-1 with no gaps"
            )

    shards = distributed.get("tensor_shards", [])
    if not isinstance(shards, list):
        errors.append("distributed.tensor_shards must be a list")
        shards = []
    groups: dict[tuple[Any, Any], list[dict]] = {}
    for shard in shards:
        if not isinstance(shard, dict):
            errors.append("distributed.tensor_shards entries must be objects")
            continue
        key = (shard.get("tensor_id"), shard.get("partition_axis"))
        groups.setdefault(key, []).append(shard)
    for (tensor_id, _axis), group in groups.items():
        ordered = sorted(group, key=lambda s: s.get("range_start", 0))
        expected_start = 0
        gap_or_overlap = False
        for shard in ordered:
            start = shard.get("range_start")
            end = shard.get("range_end")
            if not isinstance(start, int) or not isinstance(end, int) or start >= end:
                gap_or_overlap = True
                break
            if start != expected_start:
                gap_or_overlap = True
                break
            expected_start = end
        if gap_or_overlap:
            errors.append(
                f"distributed tensor_shards for '{tensor_id}' have a gap, "
                "overlap, or invalid range"
            )
        if isinstance(world_size, int) and len(group) != world_size:
            errors.append(
                f"distributed tensor_shards for '{tensor_id}' count does "
                "not match world_size (incomplete shard coverage)"
            )

    return errors


def _validate_exact_rmsnorm_decisions(payload: dict[str, Any]) -> list[str]:
    errors = []
    for function in payload.get("function_plans", []):
        for op in function.get("per_op_decisions", []):
            kernel = op.get("kernel") or {}
            if kernel.get("decision_kind") != "rmsnorm_gpu_exact_config_selection":
                continue
            for key in ("candidate_id", "operator", "semantics", "backend", "kernel_family", "kernel_entry_point", "dtype", "tokens", "hidden", "epsilon", "launch_config", "artifact", "target"):
                if kernel.get(key) is None:
                    errors.append(f"missing exact RMSNorm field: {key}")
            if kernel.get("selected_kernel") != kernel.get("candidate_id"):
                errors.append("exact RMSNorm selected_kernel must equal candidate_id")
            if kernel.get("operator") != "rmsnorm" or kernel.get("semantics") != "weighted_rmsnorm":
                errors.append("exact RMSNorm requires weighted_rmsnorm semantics")
    return errors


def _validate_aarch64_native_decisions(payload: dict[str, Any]) -> list[str]:
    errors = []
    required = ("candidate_id", "operator", "kernel_family", "dtype", "shape",
                "target", "lowering", "microkernel_id", "entry_point",
                "abi_version", "object_ref", "object_sha256",
                "backend_evidence_ref", "selection_mode",
                "selection_trace_ref", "runtime_no_redecision")
    for function in payload.get("function_plans", []):
        for op in function.get("per_op_decisions", []):
            contract = op.get("native_execution") or {}
            if contract.get("decision_kind") != "aarch64_native_exact_candidate_selection":
                continue
            for key in required:
                if contract.get(key) is None:
                    errors.append(f"missing exact AArch64 native field: {key}")
            if contract.get("operator") != "hir.fused_matmul_bias_relu":
                errors.append("AArch64 native contract operator mismatch")
            if contract.get("dtype") != "f32" or contract.get("shape") != {"m": 32, "n": 32, "k": 32}:
                errors.append("AArch64 native contract supports only f32 32x32x32")
            target, lowering = contract.get("target") or {}, contract.get("lowering") or {}
            if target.get("triple") != "aarch64-linux-gnu" or target.get("cpu") != "cortex-a76":
                errors.append("AArch64 native target contract mismatch")
            expected = {"tile_m": 8, "tile_n": 8, "tile_k": 8,
                        "vector_width_bits": 128,
                        "pipeline_id": "aarch64_tiled_scheduled_v1",
                        "loop_order_id": "tiled_mnk_row_major_v1"}
            if any(lowering.get(k) != v for k, v in expected.items()) or lowering.get("schedule_unroll_k") not in (1, 2, 4):
                errors.append("AArch64 native lowering contract mismatch")
            if contract.get("abi_version") != "mlir_ciface_memref_f32_v1":
                errors.append("AArch64 native ABI contract mismatch")
            if contract.get("runtime_no_redecision") is not True:
                errors.append("AArch64 native runtime_no_redecision must be true")
    return errors


def _contains_measured_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _MEASURED_FIELDS:
                return True
            if lowered.startswith("measured_"):
                return True
            if _contains_measured_field(child):
                return True
    elif isinstance(value, list):
        return any(_contains_measured_field(child) for child in value)
    return False

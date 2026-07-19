"""D3B Part B: typed, versioned vLLM distributed launch specification.

VLLMDistributedLaunchSpec is the structured, JSON-serializable output of the
D3B materializer (deployment/vllm_adapter/distributed_materializer.py). It
represents launch INTENT and a preflight-validated readiness classification.
It never represents a running server, a started process, or a measured
performance result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0.0"

LAUNCH_SPEC_TRUTH_BOUNDARY = (
    "A vLLM distributed launch specification is materialized launch intent "
    "derived from a compiler-selected TP plan, validated fail-closed against "
    "the actual installed vLLM version and actual host hardware. It is not a "
    "claim of successful vLLM execution, NCCL initialization, multi-GPU "
    "serving, or any measured distributed performance."
)


class FieldSource(str, Enum):
    """Provenance category for every materialized field (D3B Part C)."""

    COMPILER_PLAN = "compiler_plan"
    CAPABILITY_PROFILE = "capability_profile"
    RUNTIME_DISCOVERY = "runtime_discovery"
    EXPLICIT_D3B_DEFAULT = "explicit_D3B_default"


class WholeModelTPEvidenceStatus(str, Enum):
    NOT_ESTABLISHED_OPERATOR_LEVEL_ONLY = "not_established_operator_level_only"


class ExecutionReadinessState(str, Enum):
    """D3B may only ever reach the first three of these states."""

    MATERIALIZED = "MATERIALIZED"
    PREFLIGHT_REJECTED = "PREFLIGHT_REJECTED"
    DRY_RUN_VALIDATED = "DRY_RUN_VALIDATED"
    EXECUTION_READY = "EXECUTION_READY"       # never reached by D3B
    EXECUTION_STARTED = "EXECUTION_STARTED"   # never reached by D3B


D3B_REACHABLE_STATES = frozenset(
    {
        ExecutionReadinessState.MATERIALIZED,
        ExecutionReadinessState.PREFLIGHT_REJECTED,
        ExecutionReadinessState.DRY_RUN_VALIDATED,
    }
)


@dataclass(frozen=True)
class RankPlacement:
    rank_id: int
    logical_gpu_index: int
    physical_device_index: int | None
    rank_id_source: str
    placement_policy_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank_id": self.rank_id,
            "logical_gpu_index": self.logical_gpu_index,
            "physical_device_index": self.physical_device_index,
            "rank_id_source": self.rank_id_source,
            "placement_policy_source": self.placement_policy_source,
        }


@dataclass(frozen=True)
class FieldProvenanceEntry:
    value_summary: str
    source: FieldSource
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"value_summary": self.value_summary, "source": self.source.value, "reason": self.reason}


@dataclass(frozen=True)
class VLLMDistributedLaunchSpec:
    schema_version: str

    # Source-of-truth linkage (D2/D3A).
    source_execution_plan_id: str
    source_execution_plan_path: str
    source_candidate_id: str
    source_operator_ids: tuple[str, ...]

    # Model identity.
    model: str
    tokenizer: str
    served_model_name: str
    revision: str | None

    # Compute/precision.
    dtype: str

    # Distributed shape.
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    distributed_executor_backend: str
    world_size: int
    rank_count: int
    rank_placements: tuple[RankPlacement, ...]
    visible_devices: tuple[int, ...]
    device_type: str

    # Networking.
    host: str
    port: int
    master_address: str
    master_port: int

    # Serving/memory/batch policy.
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    trust_remote_code: bool
    seed: int

    # Generated representations (Part G/H).
    environment: dict[str, Any]
    cli_arguments: dict[str, Any]

    # Validation/readiness (Part E/F).
    preflight_status: str
    rejection_reasons: tuple[str, ...]
    execution_readiness_state: str

    # Part D: evidence-gap and mode.
    whole_model_tp_evidence_status: str
    d3b_mode: str

    # Provenance (Part C requirement: every field records its source).
    field_provenance: dict[str, FieldProvenanceEntry]

    truth_boundary: str = LAUNCH_SPEC_TRUTH_BOUNDARY

    # D4A additive extension (Part N): populated only when a validated D4A
    # whole-model evidence artifact was supplied to the materializer; None
    # for every existing call site and every D3B test, preserving identical
    # behavior when omitted.
    whole_model_tp_evidence_source_artifact_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_execution_plan_id": self.source_execution_plan_id,
            "source_execution_plan_path": self.source_execution_plan_path,
            "source_candidate_id": self.source_candidate_id,
            "source_operator_ids": list(self.source_operator_ids),
            "model": self.model,
            "tokenizer": self.tokenizer,
            "served_model_name": self.served_model_name,
            "revision": self.revision,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "distributed_executor_backend": self.distributed_executor_backend,
            "world_size": self.world_size,
            "rank_count": self.rank_count,
            "rank_placements": [r.to_dict() for r in self.rank_placements],
            "visible_devices": list(self.visible_devices),
            "device_type": self.device_type,
            "host": self.host,
            "port": self.port,
            "master_address": self.master_address,
            "master_port": self.master_port,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "trust_remote_code": self.trust_remote_code,
            "seed": self.seed,
            "environment": self.environment,
            "cli_arguments": self.cli_arguments,
            "preflight_status": self.preflight_status,
            "rejection_reasons": list(self.rejection_reasons),
            "execution_readiness_state": self.execution_readiness_state,
            "whole_model_tp_evidence_status": self.whole_model_tp_evidence_status,
            "d3b_mode": self.d3b_mode,
            "field_provenance": {k: v.to_dict() for k, v in self.field_provenance.items()},
            "truth_boundary": self.truth_boundary,
            "whole_model_tp_evidence_source_artifact_hash": self.whole_model_tp_evidence_source_artifact_hash,
        }

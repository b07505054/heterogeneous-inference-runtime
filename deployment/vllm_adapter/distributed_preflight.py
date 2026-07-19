"""D3B Part E/F: fail-closed preflight validation and execution readiness states.

D3B may only ever reach MATERIALIZED, PREFLIGHT_REJECTED, or
DRY_RUN_VALIDATED (deployment.vllm_adapter.distributed_launch_spec.
ExecutionReadinessState). EXECUTION_READY and EXECUTION_STARTED are never
assigned here -- reaching them requires an actual subprocess launch, which
is out of scope for D3B and is never attempted by this module.

No rejected launch spec ever causes a subprocess to be created: this module
performs only read-only probes (socket bind-test, filesystem cache check,
in-process introspection) and never calls subprocess.Popen/os.exec* itself.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str  # "pass" | "fail" | "advisory_fail" | "not_applicable"
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class PreflightResult:
    checks: tuple[PreflightCheck, ...]
    passed: bool
    rejection_reasons: tuple[str, ...]
    advisory_reasons: tuple[str, ...]
    primary_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "execution_preflight": "passed" if self.passed else "rejected",
            "rejection_reasons": list(self.rejection_reasons),
            "advisory_reasons": list(self.advisory_reasons),
            "primary_reason": self.primary_reason,
        }


@dataclass(frozen=True)
class PreflightInputs:
    model: str
    model_locally_resolvable: bool
    vllm_installed: bool
    all_cli_arguments_supported: bool
    unsupported_cli_arguments: tuple[str, ...]
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    world_size: int
    visible_gpu_count: int
    cuda_available: bool
    per_rank_gpu_memory_mb: tuple[float, ...]
    estimated_model_footprint_mb: float
    gpu_memory_utilization: float
    dtype: str
    supported_dtypes: tuple[str, ...]
    bf16_hardware_supported: bool | None
    max_model_len: int
    model_max_position_embeddings: int | None
    port: int
    port_available: bool
    master_address: str
    rank_placement_valid: bool
    rank_placement_errors: tuple[str, ...]
    rank_ids_contiguous: bool
    no_duplicate_physical_device: bool
    placement_count_equals_world_size: bool
    environ_conflicts: tuple[str, ...]
    distributed_executor_backend: str
    supported_executor_backends: tuple[str, ...]
    whole_model_tp_evidence_established: bool


def check_port_available(host: str, port: int) -> bool:
    """Real, read-only socket bind test -- never leaves a listening socket behind."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _hf_hub_cache_root() -> Path:
    """Resolve the real HF hub cache root exactly as huggingface_hub itself
    does (HF_HUB_CACHE, then HF_HOME/hub, then ~/.cache/huggingface/hub) --
    D4B discovered that hardcoding the default path silently mismatched a
    host with a custom HF_HOME (e.g. HF_HOME=/workspace/.hf_home), so this
    defers to the library's own resolution instead of re-implementing it."""
    try:
        from huggingface_hub import constants as hf_constants

        return Path(hf_constants.HF_HUB_CACHE)
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub"


def check_model_locally_resolvable(model_id: str) -> bool:
    """Filesystem-only check of the local HF cache -- no network call."""
    cache_root = _hf_hub_cache_root()
    slug = "models--" + model_id.replace("/", "--")
    model_dir = cache_root / slug
    if not model_dir.is_dir():
        return False
    snapshots = model_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _valid_host_or_ip(value: str) -> bool:
    if not value:
        return False
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        pass
    if value == "localhost":
        return True
    labels = value.split(".")
    return all(label and label.replace("-", "").isalnum() for label in labels)


def run_preflight(inputs: PreflightInputs) -> PreflightResult:
    checks: list[PreflightCheck] = []
    rejections: list[str] = []
    advisories: list[str] = []

    def rec(name: str, ok: bool, detail: str, *, code: str, advisory: bool = False) -> None:
        if ok:
            checks.append(PreflightCheck(name, "pass", detail))
            return
        status = "advisory_fail" if advisory else "fail"
        checks.append(PreflightCheck(name, status, detail))
        (advisories if advisory else rejections).append(code)

    rec(
        "model_identifier_resolvable",
        inputs.model_locally_resolvable,
        f"model={inputs.model!r} locally cached: {inputs.model_locally_resolvable}",
        code="model_not_locally_resolvable",
    )
    rec("vllm_package_installed", inputs.vllm_installed, "vllm import succeeded", code="vllm_not_installed")
    rec(
        "requested_cli_arguments_exist_in_installed_version",
        inputs.all_cli_arguments_supported,
        f"unsupported: {list(inputs.unsupported_cli_arguments)}",
        code="unsupported_cli_argument",
    )
    rec(
        "tensor_parallel_size_at_least_one",
        inputs.tensor_parallel_size >= 1,
        f"tensor_parallel_size={inputs.tensor_parallel_size}",
        code="tensor_parallel_size_below_one",
    )
    rec(
        "pipeline_parallel_size_at_least_one",
        inputs.pipeline_parallel_size >= 1,
        f"pipeline_parallel_size={inputs.pipeline_parallel_size}",
        code="pipeline_parallel_size_below_one",
    )
    expected_world_size = inputs.tensor_parallel_size * inputs.pipeline_parallel_size
    rec(
        "world_size_equals_tp_times_pp",
        inputs.world_size == expected_world_size,
        f"world_size={inputs.world_size} tp*pp={expected_world_size}",
        code="world_size_tp_pp_mismatch",
    )
    rec(
        "data_parallel_size_supported_configuration",
        inputs.data_parallel_size == 1,
        f"data_parallel_size={inputs.data_parallel_size} (D3B supports dp=1 only)",
        code="unsupported_data_parallel_configuration",
    )

    requested_gpu_count = inputs.tensor_parallel_size * inputs.pipeline_parallel_size
    rec(
        "cuda_available",
        inputs.cuda_available,
        f"cuda_available={inputs.cuda_available}",
        code="cuda_not_available",
    )
    rec(
        "sufficient_visible_gpu_count",
        inputs.visible_gpu_count >= requested_gpu_count,
        f"visible_gpu_count={inputs.visible_gpu_count} requested_gpu_count={requested_gpu_count}",
        code="insufficient_visible_gpu_count",
    )

    per_rank_budget_mb = [mem * inputs.gpu_memory_utilization for mem in inputs.per_rank_gpu_memory_mb]
    memory_plausible = all(mb >= inputs.estimated_model_footprint_mb for mb in per_rank_budget_mb) if per_rank_budget_mb else False
    rec(
        "per_rank_gpu_memory_plausible",
        memory_plausible,
        f"per_rank_budget_mb={per_rank_budget_mb} estimated_model_footprint_mb={inputs.estimated_model_footprint_mb}",
        code="insufficient_per_rank_gpu_memory",
    )

    rec(
        "model_dtype_supported_by_installed_vllm",
        inputs.dtype in inputs.supported_dtypes,
        f"dtype={inputs.dtype!r} supported={list(inputs.supported_dtypes)}",
        code="unsupported_dtype",
    )
    rec(
        "model_max_position_embeddings_compatible",
        inputs.model_max_position_embeddings is None or inputs.max_model_len <= inputs.model_max_position_embeddings,
        f"max_model_len={inputs.max_model_len} model_max_position_embeddings={inputs.model_max_position_embeddings}",
        code="max_model_len_exceeds_model_position_embeddings",
    )

    rec("port_in_valid_range", 1 <= inputs.port <= 65535, f"port={inputs.port}", code="invalid_port")
    rec("port_not_already_occupied", inputs.port_available, f"port={inputs.port} bind test", code="port_already_occupied")
    rec(
        "master_address_valid",
        _valid_host_or_ip(inputs.master_address),
        f"master_address={inputs.master_address!r}",
        code="invalid_master_address",
    )

    rec(
        "rank_placement_completeness",
        inputs.placement_count_equals_world_size,
        f"placement_count matches world_size={inputs.world_size}",
        code="rank_placement_incomplete",
    )
    rec(
        "rank_ids_contiguous",
        inputs.rank_ids_contiguous,
        "rank ids form contiguous 0..world_size-1 set",
        code="rank_ids_not_contiguous",
    )
    rec(
        "no_duplicate_physical_device_assignment",
        inputs.no_duplicate_physical_device,
        "no two ranks share one physical GPU index",
        code="duplicate_physical_device_assignment",
    )
    rec(
        "rank_placement_overall_valid",
        inputs.rank_placement_valid,
        f"errors={list(inputs.rank_placement_errors)}",
        code="malformed_rank_placement",
    )

    rec(
        "environment_variable_conflicts",
        len(inputs.environ_conflicts) == 0,
        f"conflicts={list(inputs.environ_conflicts)}",
        code="environment_variable_conflict",
    )
    rec(
        "distributed_executor_backend_supported",
        inputs.distributed_executor_backend in inputs.supported_executor_backends,
        f"backend={inputs.distributed_executor_backend!r} supported={list(inputs.supported_executor_backends)}",
        code="unsupported_distributed_executor_backend",
    )

    # Advisory (Part D): always recorded for TP>1, never treated as the
    # primary hardware rejection reason, but never silently omitted either.
    if inputs.tensor_parallel_size > 1:
        rec(
            "whole_model_tp_evidence_established",
            inputs.whole_model_tp_evidence_established,
            "D2/D3A prove operator-level TP correctness for one o_proj operator on one "
            "layer only; whole-model vLLM TP legality is not established by that evidence",
            code="whole_model_tp_evidence_not_established",
            advisory=True,
        )

    passed = len(rejections) == 0
    primary_reason = None
    if not passed:
        # insufficient_visible_gpu_count is the expected dominant hardware
        # reason on this one-GPU host for TP>1 -- prefer it as primary when
        # present so downstream consumers see the real hardware blocker
        # first, without hiding any other rejection (all remain listed).
        if "insufficient_visible_gpu_count" in rejections:
            primary_reason = "insufficient_visible_gpu_count"
        else:
            primary_reason = rejections[0]

    return PreflightResult(
        checks=tuple(checks),
        passed=passed,
        rejection_reasons=tuple(rejections),
        advisory_reasons=tuple(advisories),
        primary_reason=primary_reason,
    )

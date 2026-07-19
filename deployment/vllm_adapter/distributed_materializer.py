"""D3B Part C/D: materialize a VLLMDistributedLaunchSpec from the real D2/D3A
compiler-exported ExecutionPlan.

The compiler-exported plan (results/runtime_paths/distributed_d2_qwen_pipeline/
real_qwen_tp{1,2}_execution_plan.json) remains the source of truth. This
module never hand-authors a TP=2 vLLM config unrelated to that plan --
tensor_parallel_size, pipeline_parallel_size, world_size, and rank IDs are
always read from the loaded ExecutionPlan.distributed block (or its TP1
absence, which is itself the compiler's TP1 declaration per
deployment.execution_plan.schema's documented convention).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deployment.execution_plan.loader import load_execution_plan
from deployment.execution_plan.schema import ExecutionPlan
from deployment.vllm_adapter.distributed_argument_registry import check_argument
from deployment.vllm_adapter.distributed_capability_inventory import (
    EnvironmentInventory,
    discover_argument_registry,
    discover_environment,
)
from deployment.vllm_adapter.distributed_cli import CLIRepresentation, build_cli
from deployment.vllm_adapter.distributed_dry_run import DryRunResult, run_dry_run_validation
from deployment.vllm_adapter.distributed_environment import materialize_environment
from deployment.vllm_adapter.distributed_launch_spec import (
    SCHEMA_VERSION,
    D3B_REACHABLE_STATES,
    ExecutionReadinessState,
    FieldProvenanceEntry,
    FieldSource,
    RankPlacement,
    VLLMDistributedLaunchSpec,
    WholeModelTPEvidenceStatus,
)
from deployment.vllm_adapter.distributed_preflight import (
    PreflightInputs,
    PreflightResult,
    check_model_locally_resolvable,
    check_port_available,
    run_preflight,
)
from deployment.vllm_adapter.distributed_rank_placement import build_rank_placement

# The real HF model identity. D2's ExecutionPlan.model_identity.model_id is
# the compiler's abbreviated identifier ("qwen2.5-0.5b"); D3A independently
# validated that this identifier corresponds to the real, locally-cached
# Qwen/Qwen2.5-0.5B-Instruct checkpoint (see
# results/runtime_paths/distributed_d3a_live_qwen_tensor/live_model_execution.json).
# This mapping is part of the compiler_plan provenance chain (D2 -> D3A), not
# a fresh D3B invention.
REAL_HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
D3A_SEED = 1234  # reused verbatim from D3A's live_model_execution.json

D2_RESULTS_DIR_NAME = "distributed_d2_qwen_pipeline"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_MAX_NUM_SEQS = 4
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_DTYPE = "float16"
DEFAULT_EXECUTOR_BACKEND = "mp"

KV_CACHE_HEADROOM_MB = 512.0  # conservative fixed overhead margin for the memory-plausibility check

# D1/D2 implement exactly one distributed strategy end to end. Any other
# declared strategy must be rejected explicitly here, mirroring
# deployment/execution_plan/loader.py's KNOWN_COLLECTIVE_KINDS fail-closed
# pattern for collective "kind" (which does not itself cover top-level
# "strategy").
KNOWN_DISTRIBUTED_STRATEGIES = frozenset({"tensor_parallel"})


class UnknownDistributedStrategyError(ValueError):
    """Raised when ExecutionPlan.distributed.strategy is not a D3B-known strategy."""


@dataclass(frozen=True)
class MaterializationBundle:
    """Everything produced for one plan: the spec plus every intermediate artifact."""

    spec: VLLMDistributedLaunchSpec
    environment_inventory: EnvironmentInventory
    argument_registry: dict[str, Any]
    rank_placement: Any
    environment_materialization: Any
    cli: CLIRepresentation
    preflight: PreflightResult
    dry_run: DryRunResult
    selected_candidate_id: str
    plan: ExecutionPlan


def _effective_distributed_fields(plan: ExecutionPlan) -> tuple[int, int, int, tuple[int, ...]]:
    """Return (tensor_parallel_size, pipeline_parallel_size, world_size, rank_ids).

    TP1 plans carry no 'distributed' block at all -- that absence IS the
    compiler's TP1 declaration (see deployment/execution_plan/schema.py).
    """
    if plan.distributed is None:
        return 1, 1, 1, (0,)
    d = plan.distributed
    if d.strategy not in KNOWN_DISTRIBUTED_STRATEGIES:
        raise UnknownDistributedStrategyError(
            f"ExecutionPlan.distributed.strategy={d.strategy!r} is not a D3B-known strategy "
            f"(known: {sorted(KNOWN_DISTRIBUTED_STRATEGIES)}); refusing to materialize rather "
            "than silently guessing a launch configuration for it."
        )
    rank_ids = tuple(sorted(r.rank_id for r in d.ranks))
    return d.tensor_parallel_size, d.pipeline_parallel_size, d.world_size, rank_ids


def _load_candidate_id(d2_results_dir: Path, world_size: int) -> str:
    selection_path = d2_results_dir / "qwen_distributed_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if world_size > 1:
            return str(selection.get("selected_candidate_id", "tp2"))
    return "tp1" if world_size == 1 else "tp2"


def _source_operator_ids(plan: ExecutionPlan) -> tuple[str, ...]:
    if plan.distributed is None:
        return ()
    return tuple(sorted({s.tensor_id for s in plan.distributed.tensor_shards}))


def _estimate_model_footprint_mb() -> float:
    """Real measured size of the cached checkpoint file(s) -- not a param-count guess."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + REAL_HF_MODEL_ID.replace("/", "--")
    model_dir = cache_root / slug
    total_bytes = 0
    snapshots = model_dir / "snapshots"
    if snapshots.is_dir():
        for snap in snapshots.iterdir():
            for f in snap.glob("*.safetensors"):
                try:
                    total_bytes += f.resolve().stat().st_size
                except OSError:
                    continue
    if total_bytes == 0:
        return 1200.0  # documented conservative fallback if cache layout ever changes
    return total_bytes / (1024 * 1024) + KV_CACHE_HEADROOM_MB


def _model_max_position_embeddings() -> int | None:
    try:
        from transformers import AutoConfig  # noqa: PLC0415

        cfg = AutoConfig.from_pretrained(REAL_HF_MODEL_ID)
        return int(cfg.max_position_embeddings)
    except Exception:  # noqa: BLE001 -- offline/unavailable config must not crash materialization
        return None


def _environ_conflicts(planned: dict[str, str]) -> tuple[str, ...]:
    conflicts = []
    for key, value in planned.items():
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            conflicts.append(f"{key}: process environment already sets {existing!r}, planned {value!r}")
    return tuple(conflicts)


def _resolve_whole_model_tp_evidence(
    d4a_evidence_path: str | Path | None, *, tensor_parallel_size: int,
) -> tuple[str, str | None]:
    """D4A Part N: additive, backward-compatible evidence upgrade.

    Returns (whole_model_tp_evidence_status, source_artifact_hash). Absent,
    unreadable, or non-matching evidence always falls back to D3B's
    original status -- this never raises and never silently upgrades on
    bad input (Part O: "D3B whole-model evidence updated without D4A
    artifact" must not happen).
    """
    default_status = WholeModelTPEvidenceStatus.NOT_ESTABLISHED_OPERATOR_LEVEL_ONLY.value
    if d4a_evidence_path is None or tensor_parallel_size < 2:
        return default_status, None
    path = Path(d4a_evidence_path)
    if not path.exists():
        return default_status, None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default_status, None
    if payload.get("classification") != "WHOLE_MODEL_TP_VALIDATED":
        return default_status, None
    if payload.get("model") != REAL_HF_MODEL_ID or payload.get("tensor_parallel_size") != 2:
        return default_status, None
    artifact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return "validated_serialized_whole_model_contract", artifact_hash


def materialize_launch_spec(
    execution_plan_path: str | Path,
    *,
    repo_root: Path,
    d3b_mode: str = "planning_only",
    d4a_evidence_path: str | Path | None = None,
) -> MaterializationBundle:
    if d3b_mode != "planning_only":
        raise ValueError("D3B only supports d3b_mode='planning_only'; no execution mode exists here")

    plan_path = Path(execution_plan_path)
    plan = load_execution_plan(plan_path)
    tp, pp, world_size, rank_ids = _effective_distributed_fields(plan)

    d2_results_dir = repo_root / "results" / "runtime_paths" / D2_RESULTS_DIR_NAME
    candidate_id = _load_candidate_id(d2_results_dir, world_size)
    operator_ids = _source_operator_ids(plan)

    env_inv = discover_environment()
    registry = discover_argument_registry()

    rank_placement = build_rank_placement(
        compiler_rank_ids=rank_ids, world_size=world_size, visible_gpu_count=env_inv.visible_gpu_count
    )

    whole_model_tp_evidence_established = False  # D3B never establishes this; see Part D

    field_provenance: dict[str, FieldProvenanceEntry] = {}

    def prov(name: str, value: Any, source: FieldSource, reason: str) -> Any:
        field_provenance[name] = FieldProvenanceEntry(value_summary=repr(value)[:200], source=source, reason=reason)
        return value

    served_model_name = prov(
        "served_model_name",
        f"{REAL_HF_MODEL_ID.split('/')[-1].lower()}-{candidate_id}-planning-only",
        FieldSource.EXPLICIT_D3B_DEFAULT,
        "D2/D3A plans do not declare a served name; D3B names it after the model and candidate id.",
    )
    model = prov(
        "model",
        REAL_HF_MODEL_ID,
        FieldSource.COMPILER_PLAN,
        "D3A validated that the compiler's abbreviated model_identity.model_id "
        f"({plan.model_identity.get('model_id')!r}) corresponds to this real, locally-cached HF checkpoint.",
    )
    tokenizer = prov("tokenizer", REAL_HF_MODEL_ID, FieldSource.COMPILER_PLAN, "Same identity as model; no separate compiler tokenizer decision exists.")
    revision = prov("revision", None, FieldSource.CAPABILITY_PROFILE, "Matches installed vLLM registry default for --revision (None).")
    dtype = prov(
        "dtype", DEFAULT_DTYPE, FieldSource.EXPLICIT_D3B_DEFAULT,
        "Neither D2 nor D3A declares a serving dtype for the distributed plan; float16 is pinned "
        "explicitly for deterministic, reproducible CLI generation rather than leaving 'auto' "
        "(the installed registry default) to resolve differently across runs.",
    )
    tensor_parallel_size = prov("tensor_parallel_size", tp, FieldSource.COMPILER_PLAN, "Read directly from ExecutionPlan.distributed.tensor_parallel_size (or the TP1 absence convention).")
    pipeline_parallel_size = prov("pipeline_parallel_size", pp, FieldSource.COMPILER_PLAN, "Read directly from ExecutionPlan.distributed.pipeline_parallel_size (or the TP1 absence convention).")
    data_parallel_size = prov("data_parallel_size", 1, FieldSource.CAPABILITY_PROFILE, "Matches installed vLLM registry default for --data-parallel-size (1); D2 plan carries no data-parallel concept.")
    distributed_executor_backend = prov(
        "distributed_executor_backend", DEFAULT_EXECUTOR_BACKEND, FieldSource.EXPLICIT_D3B_DEFAULT,
        "D2 plan declares a tensor-parallel strategy but not a vLLM executor backend choice; "
        "'mp' (multiprocessing) is picked as the single-node default over 'ray'/'external_launcher'/'uni'.",
    )
    world_size_val = prov("world_size", world_size, FieldSource.COMPILER_PLAN, "Read directly from ExecutionPlan.distributed.world_size (or the TP1 absence convention).")
    rank_count = prov("rank_count", len(rank_placement.placements), FieldSource.COMPILER_PLAN, "Equal to the number of compiler-declared ranks.")
    visible_devices = prov(
        "visible_devices", tuple(range(env_inv.visible_gpu_count)), FieldSource.RUNTIME_DISCOVERY,
        "Directly probed via torch.cuda.device_count() on this host at materialization time.",
    )
    device_type = prov("device_type", "cuda" if env_inv.cuda_available else "cpu", FieldSource.RUNTIME_DISCOVERY, "torch.cuda.is_available() probed at materialization time.")
    host = prov("host", DEFAULT_HOST, FieldSource.EXPLICIT_D3B_DEFAULT, "Installed registry default for --host is None (bind-all); D3B pins loopback explicitly for a safe planning default.")
    port = prov("port", DEFAULT_PORT, FieldSource.CAPABILITY_PROFILE, "Matches installed vLLM registry default for --port (8000).")
    master_address = prov("master_address", "127.0.0.1", FieldSource.CAPABILITY_PROFILE, "Matches installed vLLM registry default for --master-addr.")
    master_port = prov("master_port", 29501, FieldSource.CAPABILITY_PROFILE, "Matches installed vLLM registry default for --master-port.")
    max_model_len = prov(
        "max_model_len", DEFAULT_MAX_MODEL_LEN, FieldSource.EXPLICIT_D3B_DEFAULT,
        "Neither D2 nor D3A declares a serving context length; 2048 is a conservative D3B default "
        "well within the real model's max_position_embeddings.",
    )
    max_num_seqs = prov("max_num_seqs", DEFAULT_MAX_NUM_SEQS, FieldSource.EXPLICIT_D3B_DEFAULT, "Conservative default batch width appropriate for a 4GB-class GPU; not decided by the compiler plan.")
    max_num_batched_tokens = prov("max_num_batched_tokens", DEFAULT_MAX_NUM_BATCHED_TOKENS, FieldSource.EXPLICIT_D3B_DEFAULT, "Conservative default token budget per step; not decided by the compiler plan.")
    gpu_memory_utilization = prov(
        "gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION, FieldSource.EXPLICIT_D3B_DEFAULT,
        "Set below the installed registry default (0.92) given the small 4GB-class GPU on this host.",
    )
    enable_prefix_caching = prov("enable_prefix_caching", True, FieldSource.EXPLICIT_D3B_DEFAULT, "Registry default leaves this unset (None/auto); D3B explicitly enables it.")
    enable_chunked_prefill = prov("enable_chunked_prefill", True, FieldSource.EXPLICIT_D3B_DEFAULT, "Registry default leaves this unset (None/auto); D3B explicitly enables it.")
    trust_remote_code = prov("trust_remote_code", False, FieldSource.CAPABILITY_PROFILE, "Matches installed vLLM registry default (False); real Qwen2.5 config does not require custom remote code.")
    seed = prov("seed", D3A_SEED, FieldSource.COMPILER_PLAN, "Reused verbatim from D3A's live_model_execution.json seed, preserving determinism continuity across D2/D3A/D3B.")

    env_mat = materialize_environment(
        visible_physical_devices=tuple(p.physical_device_index for p in rank_placement.placements if p.physical_device_index is not None),
        master_address=master_address,
        master_port=master_port,
        world_size=world_size_val,
        distributed_executor_backend=distributed_executor_backend,
    )
    environment = env_mat.as_flat_map()

    fields_for_cli = {
        "model": model, "tokenizer": tokenizer, "trust_remote_code": trust_remote_code,
        "dtype": dtype, "seed": seed, "revision": revision, "served_model_name": served_model_name,
        "host": host, "port": port, "master_address": master_address, "master_port": master_port,
        "tensor_parallel_size": tensor_parallel_size, "pipeline_parallel_size": pipeline_parallel_size,
        "data_parallel_size": data_parallel_size, "distributed_executor_backend": distributed_executor_backend,
        "max_model_len": max_model_len, "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens, "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": enable_prefix_caching, "enable_chunked_prefill": enable_chunked_prefill,
        "world_size": world_size_val,
        "_sources": {k: v.source.value for k, v in field_provenance.items()},
    }
    cli = build_cli(
        fields_for_cli,
        registry=registry,
        environment=environment,
        working_directory=str(repo_root),
        rank_placements=[p.to_dict() for p in rank_placement.placements],
    )
    # Command reproducibility check (Part K): rebuild once more from the same inputs.
    cli_repeat = build_cli(
        fields_for_cli,
        registry=registry,
        environment=environment,
        working_directory=str(repo_root),
        rank_placements=[p.to_dict() for p in rank_placement.placements],
    )

    model_locally_resolvable = check_model_locally_resolvable(REAL_HF_MODEL_ID)
    port_available = check_port_available(host, port)
    estimated_footprint_mb = _estimate_model_footprint_mb()
    max_position_embeddings = _model_max_position_embeddings()
    per_rank_memory = tuple(g.total_memory_mb for g in env_inv.gpus)

    dtype_spec = registry.get("arguments", {}).get("dtype", {})
    supported_dtypes = tuple(dtype_spec.get("choices") or ())
    backend_spec = registry.get("arguments", {}).get("distributed_executor_backend", {})
    supported_backends: tuple[str, ...] = ()
    if backend_spec.get("choices_from_metavar"):
        import ast

        try:
            supported_backends = tuple(ast.literal_eval(backend_spec["choices_from_metavar"]))
        except (ValueError, SyntaxError):
            supported_backends = ()

    preflight_inputs = PreflightInputs(
        model=model,
        model_locally_resolvable=model_locally_resolvable,
        vllm_installed=env_inv.vllm_version is not None,
        all_cli_arguments_supported=cli.all_arguments_supported,
        unsupported_cli_arguments=cli.unsupported_arguments,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        world_size=world_size_val,
        visible_gpu_count=env_inv.visible_gpu_count,
        cuda_available=env_inv.cuda_available,
        per_rank_gpu_memory_mb=per_rank_memory,
        estimated_model_footprint_mb=estimated_footprint_mb,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        supported_dtypes=supported_dtypes,
        bf16_hardware_supported=env_inv.bf16_supported,
        max_model_len=max_model_len,
        model_max_position_embeddings=max_position_embeddings,
        port=port,
        port_available=port_available,
        master_address=master_address,
        rank_placement_valid=rank_placement.valid,
        rank_placement_errors=rank_placement.errors,
        rank_ids_contiguous=rank_placement.rank_ids_contiguous,
        no_duplicate_physical_device=rank_placement.no_duplicate_physical_device,
        placement_count_equals_world_size=rank_placement.placement_count_equals_world_size,
        environ_conflicts=_environ_conflicts(environment),
        distributed_executor_backend=distributed_executor_backend,
        supported_executor_backends=supported_backends,
        whole_model_tp_evidence_established=whole_model_tp_evidence_established,
    )
    preflight = run_preflight(preflight_inputs)

    spec_dict_preview = {
        "schema_version": SCHEMA_VERSION,
        "source_execution_plan_id": plan.plan_id,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "world_size": world_size_val,
        "rank_placements": [p.to_dict() for p in rank_placement.placements],
        "preflight_status": "passed" if preflight.passed else "rejected",
        "execution_readiness_state": "placeholder",
        "whole_model_tp_evidence_status": WholeModelTPEvidenceStatus.NOT_ESTABLISHED_OPERATOR_LEVEL_ONLY.value,
        "truth_boundary": "placeholder",
        "port": port,
    }

    dry_run = run_dry_run_validation(
        argv=cli.argv,
        spec_dict=spec_dict_preview,
        environment=environment,
        world_size=world_size_val,
        rank_placement_count=len(rank_placement.placements),
        reproduced_argv=cli_repeat.argv,
    )

    if not preflight.passed:
        readiness_state = ExecutionReadinessState.PREFLIGHT_REJECTED
    elif dry_run.passed:
        readiness_state = ExecutionReadinessState.DRY_RUN_VALIDATED
    else:
        readiness_state = ExecutionReadinessState.MATERIALIZED
    assert readiness_state in D3B_REACHABLE_STATES

    whole_model_evidence_status, whole_model_evidence_hash = _resolve_whole_model_tp_evidence(
        d4a_evidence_path, tensor_parallel_size=tensor_parallel_size,
    )

    spec = VLLMDistributedLaunchSpec(
        schema_version=SCHEMA_VERSION,
        source_execution_plan_id=plan.plan_id,
        source_execution_plan_path=str(plan_path),
        source_candidate_id=candidate_id,
        source_operator_ids=operator_ids,
        model=model,
        tokenizer=tokenizer,
        served_model_name=served_model_name,
        revision=revision,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        world_size=world_size_val,
        rank_count=rank_count,
        rank_placements=rank_placement.placements,
        visible_devices=visible_devices,
        device_type=device_type,
        host=host,
        port=port,
        master_address=master_address,
        master_port=master_port,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        trust_remote_code=trust_remote_code,
        seed=seed,
        environment=environment,
        cli_arguments=cli.to_dict(),
        preflight_status="passed" if preflight.passed else "rejected",
        rejection_reasons=preflight.rejection_reasons,
        execution_readiness_state=readiness_state.value,
        whole_model_tp_evidence_status=whole_model_evidence_status,
        d3b_mode=d3b_mode,
        field_provenance=field_provenance,
        whole_model_tp_evidence_source_artifact_hash=whole_model_evidence_hash,
    )

    return MaterializationBundle(
        spec=spec,
        environment_inventory=env_inv,
        argument_registry=registry,
        rank_placement=rank_placement,
        environment_materialization=env_mat,
        cli=cli,
        preflight=preflight,
        dry_run=dry_run,
        selected_candidate_id=candidate_id,
        plan=plan,
    )

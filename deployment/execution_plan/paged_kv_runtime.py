"""Construct live paged-KV runtime objects from compiler ExecutionPlan contracts."""
from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deployment.execution_plan.kv_page_manager import KVPageManager
from deployment.execution_plan.paged_kv_cache import (
    PagedKVAttentionSession,
    PagedKVStorage,
)
from deployment.execution_plan.schema import ExecutionPlan, PagedKVExecutionContract
from deployment.serving_scheduler import ReplicaSchedulerState, SchedulerProfile


class PagedKVRuntimeContractError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class PagedKVRuntimeContext:
    contract: PagedKVExecutionContract
    artifact_root: Path
    page_manager: KVPageManager
    storage: PagedKVStorage

    def create_session(self, request_id: str) -> PagedKVAttentionSession:
        return PagedKVAttentionSession(
            self.contract.to_session_contract(),
            artifact_root=self.artifact_root,
            request_id=request_id,
            storage=self.storage,
            page_manager=self.page_manager,
        )

    def scheduler_state(
        self,
        replica_id: str,
        profile: SchedulerProfile,
    ) -> ReplicaSchedulerState:
        return ReplicaSchedulerState(
            replica_id,
            profile,
            page_manager=self.page_manager,
        )


def paged_kv_contracts(plan: ExecutionPlan) -> tuple[PagedKVExecutionContract, ...]:
    contracts: list[PagedKVExecutionContract] = []
    for function in plan.function_plans:
        for op in function.per_op_decisions:
            if op.paged_kv_execution is not None:
                contracts.append(op.paged_kv_execution)
    return tuple(contracts)


def build_paged_kv_runtime(
    contract_or_plan: PagedKVExecutionContract | ExecutionPlan,
    artifact_root: str | Path,
) -> PagedKVRuntimeContext:
    contract = _single_contract(contract_or_plan)
    _validate_runtime_no_redecision(contract)
    artifact_root = Path(artifact_root)
    _validate_artifact_and_entry_points(contract, artifact_root)

    page_manager = KVPageManager(
        total_pages=contract.num_physical_pages,
        tokens_per_page=contract.page_tokens,
    )
    storage = PagedKVStorage(
        total_pages=contract.num_physical_pages,
        num_kv_heads=contract.num_kv_heads,
        tokens_per_page=contract.page_tokens,
        head_dim=contract.head_dim,
        dtype=contract.dtype,
        workspace_tokens=contract.maximum_logical_tokens,
    )
    if page_manager.total_pages != storage.total_pages:
        raise PagedKVRuntimeContractError("manager_storage_page_count_mismatch")
    if page_manager.tokens_per_page != storage.tokens_per_page:
        raise PagedKVRuntimeContractError("manager_storage_tokens_per_page_mismatch")
    return PagedKVRuntimeContext(
        contract=contract,
        artifact_root=artifact_root,
        page_manager=page_manager,
        storage=storage,
    )


def _single_contract(
    contract_or_plan: PagedKVExecutionContract | ExecutionPlan,
) -> PagedKVExecutionContract:
    if isinstance(contract_or_plan, PagedKVExecutionContract):
        return contract_or_plan
    contracts = paged_kv_contracts(contract_or_plan)
    if not contracts:
        raise PagedKVRuntimeContractError("missing_paged_kv_execution_contract")
    first = contracts[0]
    if any(_runtime_identity(contract) != _runtime_identity(first)
           for contract in contracts[1:]):
        raise PagedKVRuntimeContractError("multiple_distinct_paged_kv_contracts")
    return first


def _runtime_identity(contract: PagedKVExecutionContract) -> dict[str, Any]:
    ignored = {"operation_order", "truth_boundary"}
    return {k: v for k, v in contract.raw.items() if k not in ignored}


def _validate_runtime_no_redecision(contract: PagedKVExecutionContract) -> None:
    if not contract.runtime_no_layout_redecision:
        raise PagedKVRuntimeContractError("runtime_layout_redecision_not_supported")
    if not contract.runtime_no_kernel_redecision:
        raise PagedKVRuntimeContractError("runtime_kernel_redecision_not_supported")


def _validate_artifact_and_entry_points(
    contract: PagedKVExecutionContract,
    artifact_root: Path,
) -> None:
    artifact = artifact_root / contract.pool_artifact_ref
    if not artifact.is_file():
        raise PagedKVRuntimeContractError(f"paged_kv_artifact_not_found:{artifact}")
    if _sha256(artifact) != contract.pool_artifact_sha256:
        raise PagedKVRuntimeContractError("paged_kv_artifact_hash_mismatch")

    lib = ctypes.CDLL(str(artifact))
    version = getattr(lib, "hir_paged_kv_artifact_version", None)
    if version is None:
        raise PagedKVRuntimeContractError("missing_entry_point:hir_paged_kv_artifact_version")
    version.restype = ctypes.c_char_p
    if version().decode() != contract.pool_artifact_version:
        raise PagedKVRuntimeContractError("paged_kv_artifact_version_mismatch")

    for entry_point in (
        contract.pool_create_entry_point,
        contract.prefill_write_entry_point,
        contract.append_entry_point,
        contract.reset_entry_point,
        contract.paged_attention_entry_point,
    ):
        if not hasattr(lib, entry_point):
            raise PagedKVRuntimeContractError(f"missing_entry_point:{entry_point}")


def build_paged_kv_runtime_from_payload(
    payload: dict[str, Any],
    artifact_root: str | Path,
) -> PagedKVRuntimeContext:
    from deployment.execution_plan.loader import parse_execution_plan

    return build_paged_kv_runtime(parse_execution_plan(payload), artifact_root)

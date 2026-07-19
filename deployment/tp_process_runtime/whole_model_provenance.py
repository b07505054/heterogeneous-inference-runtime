"""D4A Part P: cross-layer provenance counters.

Every counter is COMPUTED from the actual inventory/mapping/shard/
comparison results produced by this run -- never hardcoded to zero. A
successful D4A run requires every counter to equal zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WholeModelProvenanceCounters:
    operator_inventory_mismatch_count: int
    vllm_contract_mismatch_count: int
    compiler_mapping_mismatch_count: int
    weight_shard_mismatch_count: int
    activation_shard_mismatch_count: int
    head_partition_mismatch_count: int
    kv_partition_mismatch_count: int
    collective_mismatch_count: int
    bias_mismatch_count: int
    vocab_partition_mismatch_count: int
    replicated_boundary_mismatch_count: int
    block_output_mismatch_count: int
    whole_model_output_mismatch_count: int
    synthetic_fallback_count: int
    full_operator_bypass_count: int
    temporary_tensor_leak_count: int
    orphan_process_count: int

    def to_dict(self) -> dict[str, Any]:
        d = {
            "operator_inventory_mismatch_count": self.operator_inventory_mismatch_count,
            "vllm_contract_mismatch_count": self.vllm_contract_mismatch_count,
            "compiler_mapping_mismatch_count": self.compiler_mapping_mismatch_count,
            "weight_shard_mismatch_count": self.weight_shard_mismatch_count,
            "activation_shard_mismatch_count": self.activation_shard_mismatch_count,
            "head_partition_mismatch_count": self.head_partition_mismatch_count,
            "kv_partition_mismatch_count": self.kv_partition_mismatch_count,
            "collective_mismatch_count": self.collective_mismatch_count,
            "bias_mismatch_count": self.bias_mismatch_count,
            "vocab_partition_mismatch_count": self.vocab_partition_mismatch_count,
            "replicated_boundary_mismatch_count": self.replicated_boundary_mismatch_count,
            "block_output_mismatch_count": self.block_output_mismatch_count,
            "whole_model_output_mismatch_count": self.whole_model_output_mismatch_count,
            "synthetic_fallback_count": self.synthetic_fallback_count,
            "full_operator_bypass_count": self.full_operator_bypass_count,
            "temporary_tensor_leak_count": self.temporary_tensor_leak_count,
            "orphan_process_count": self.orphan_process_count,
        }
        d["all_zero"] = all(v == 0 for v in d.values())
        return d

    def all_zero(self) -> bool:
        return all(
            v == 0
            for v in (
                self.operator_inventory_mismatch_count, self.vllm_contract_mismatch_count,
                self.compiler_mapping_mismatch_count, self.weight_shard_mismatch_count,
                self.activation_shard_mismatch_count, self.head_partition_mismatch_count,
                self.kv_partition_mismatch_count, self.collective_mismatch_count,
                self.bias_mismatch_count, self.vocab_partition_mismatch_count,
                self.replicated_boundary_mismatch_count, self.block_output_mismatch_count,
                self.whole_model_output_mismatch_count, self.synthetic_fallback_count,
                self.full_operator_bypass_count, self.temporary_tensor_leak_count,
                self.orphan_process_count,
            )
        )


def compute_provenance_counters(
    *,
    operator_records: list[Any],
    vllm_contract_facts: dict[str, Any],
    compiler_mapping_results: list[dict[str, Any]],
    shard_coverage_errors: list[str],
    activation_shard_errors: list[str],
    head_partition_checks: dict[str, bool],
    kv_partition_checks: dict[str, bool],
    collective_kinds_seen: list[str],
    known_collective_kinds: frozenset,
    bias_contract_checks: dict[str, bool],
    vocab_partition_checks: dict[str, bool],
    replicated_boundary_checks: dict[str, bool],
    block_max_abs_errors: dict[str, float],
    block_tolerance_atol: float,
    whole_model_logits_max_abs_error: float,
    whole_model_logits_tolerance_atol: float,
    whole_model_argmax_match: bool,
    whole_model_topk_match: bool,
    synthetic_fallback_events: int,
    full_operator_bypass_events: int,
    temp_leak_candidates_found: int,
    orphan_pids_found: list[int],
) -> WholeModelProvenanceCounters:
    operator_inventory_mismatch_count = sum(
        1 for r in operator_records if r.validation_status not in ("validated", "validated_replicated")
    )
    vllm_contract_mismatch_count = sum(
        1 for fact in vllm_contract_facts.values()
        if not fact.get("verification_excerpt") or not fact.get("source_sha256")
    )
    compiler_mapping_mismatch_count = sum(1 for r in compiler_mapping_results if not r.get("mapped_ok"))
    weight_shard_mismatch_count = len(shard_coverage_errors)
    activation_shard_mismatch_count = len(activation_shard_errors)
    head_partition_mismatch_count = sum(1 for ok in head_partition_checks.values() if not ok)
    kv_partition_mismatch_count = sum(1 for ok in kv_partition_checks.values() if not ok)
    collective_mismatch_count = sum(1 for k in collective_kinds_seen if k not in known_collective_kinds)
    bias_mismatch_count = sum(1 for ok in bias_contract_checks.values() if not ok)
    vocab_partition_mismatch_count = sum(1 for ok in vocab_partition_checks.values() if not ok)
    replicated_boundary_mismatch_count = sum(1 for ok in replicated_boundary_checks.values() if not ok)
    block_output_mismatch_count = sum(1 for err in block_max_abs_errors.values() if err > block_tolerance_atol)
    whole_model_output_mismatch_count = int(
        whole_model_logits_max_abs_error > whole_model_logits_tolerance_atol
        or not whole_model_argmax_match or not whole_model_topk_match
    )

    return WholeModelProvenanceCounters(
        operator_inventory_mismatch_count=operator_inventory_mismatch_count,
        vllm_contract_mismatch_count=vllm_contract_mismatch_count,
        compiler_mapping_mismatch_count=compiler_mapping_mismatch_count,
        weight_shard_mismatch_count=weight_shard_mismatch_count,
        activation_shard_mismatch_count=activation_shard_mismatch_count,
        head_partition_mismatch_count=head_partition_mismatch_count,
        kv_partition_mismatch_count=kv_partition_mismatch_count,
        collective_mismatch_count=collective_mismatch_count,
        bias_mismatch_count=bias_mismatch_count,
        vocab_partition_mismatch_count=vocab_partition_mismatch_count,
        replicated_boundary_mismatch_count=replicated_boundary_mismatch_count,
        block_output_mismatch_count=block_output_mismatch_count,
        whole_model_output_mismatch_count=whole_model_output_mismatch_count,
        synthetic_fallback_count=synthetic_fallback_events,
        full_operator_bypass_count=full_operator_bypass_events,
        temporary_tensor_leak_count=temp_leak_candidates_found,
        orphan_process_count=len(orphan_pids_found),
    )

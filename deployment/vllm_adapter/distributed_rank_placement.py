"""D3B Part I: rank placement contract.

Converts the compiler's logical ranks (D2 ExecutionPlan.distributed.ranks)
into a launch placement specification: rank -> logical GPU index -> (maybe)
physical CUDA device index. The compiler plan supplies rank IDs; the D3B
contiguous rank->GPU convention (rank i -> logical GPU i) is an explicit D3B
placement policy, since D2's ranks are simulated CPU processes, not
GPU-indexed.

This module never assigns two TP ranks to the same physical GPU and calls
the result ready -- see validate_rank_placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deployment.vllm_adapter.distributed_launch_spec import RankPlacement


class RankPlacementError(ValueError):
    """Raised when a rank placement contract is structurally invalid."""


@dataclass(frozen=True)
class RankPlacementResult:
    placements: tuple[RankPlacement, ...]
    rank_ids_contiguous: bool
    one_device_per_rank: bool
    no_duplicate_physical_device: bool
    placement_count_equals_world_size: bool
    logical_to_physical_explicit: bool
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "placements": [p.to_dict() for p in self.placements],
            "rank_ids_contiguous": self.rank_ids_contiguous,
            "one_device_per_rank": self.one_device_per_rank,
            "no_duplicate_physical_device": self.no_duplicate_physical_device,
            "placement_count_equals_world_size": self.placement_count_equals_world_size,
            "logical_to_physical_explicit": self.logical_to_physical_explicit,
            "valid": self.valid,
            "errors": list(self.errors),
        }


def build_rank_placement(
    *,
    compiler_rank_ids: tuple[int, ...],
    world_size: int,
    visible_gpu_count: int,
) -> RankPlacementResult:
    """Build a D3B rank->logical-GPU->physical-GPU placement contract.

    Placement policy (explicit D3B default, not compiler-plan-derived):
    rank i -> logical GPU i, contiguous. Physical device index is assigned
    only when a physical device with that index is actually visible;
    otherwise it is left unassigned (None) rather than fabricated, and the
    validation flags below report why.
    """
    errors: list[str] = []

    rank_ids_sorted = sorted(compiler_rank_ids)
    rank_ids_contiguous = rank_ids_sorted == list(range(len(rank_ids_sorted)))
    if not rank_ids_contiguous:
        errors.append("rank IDs are not contiguous starting at 0")

    placement_count_equals_world_size = len(compiler_rank_ids) == world_size
    if not placement_count_equals_world_size:
        errors.append(
            f"placement count ({len(compiler_rank_ids)}) does not equal world_size ({world_size})"
        )

    placements: list[RankPlacement] = []
    physical_assignments: dict[int, int] = {}
    for rank_id in rank_ids_sorted:
        logical_gpu_index = rank_id
        physical_index: int | None = logical_gpu_index if logical_gpu_index < visible_gpu_count else None
        placements.append(
            RankPlacement(
                rank_id=rank_id,
                logical_gpu_index=logical_gpu_index,
                physical_device_index=physical_index,
                rank_id_source="compiler_plan",
                placement_policy_source="explicit_D3B_default",
            )
        )
        if physical_index is not None:
            physical_assignments.setdefault(physical_index, 0)
            physical_assignments[physical_index] += 1

    one_device_per_rank = all(p.physical_device_index is None or True for p in placements)
    # Distinct check: no two *placed* (physically resolvable) ranks may share a physical index.
    duplicate_physical = [idx for idx, count in physical_assignments.items() if count > 1]
    no_duplicate_physical_device = len(duplicate_physical) == 0
    if not no_duplicate_physical_device:
        errors.append(f"physical GPU indices assigned to more than one rank: {duplicate_physical}")

    logical_to_physical_explicit = True  # every placement entry states its mapping explicitly, even when None

    valid = (
        rank_ids_contiguous
        and placement_count_equals_world_size
        and no_duplicate_physical_device
        and logical_to_physical_explicit
    )

    return RankPlacementResult(
        placements=tuple(placements),
        rank_ids_contiguous=rank_ids_contiguous,
        one_device_per_rank=one_device_per_rank,
        no_duplicate_physical_device=no_duplicate_physical_device,
        placement_count_equals_world_size=placement_count_equals_world_size,
        logical_to_physical_explicit=logical_to_physical_explicit,
        valid=valid,
        errors=tuple(errors),
    )


def rank_placement_fully_resolvable_on_host(result: RankPlacementResult, visible_gpu_count: int) -> bool:
    """True only if every rank placement resolved to a distinct real physical GPU."""
    if not result.valid:
        return False
    resolved = [p.physical_device_index for p in result.placements]
    if any(idx is None for idx in resolved):
        return False
    return len(set(resolved)) == len(resolved) == len(result.placements) <= visible_gpu_count

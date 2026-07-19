"""D3B Part H: environment materialization.

Only variables that D3B's own materialized launch actually needs are
included -- nothing is fabricated for variables vLLM manages internally
(e.g. vLLM's OpenAI server does not require the caller to set RANK/
LOCAL_RANK/WORLD_SIZE by hand; its own multiproc executor assigns those to
its worker subprocesses). Every included variable records why it is present;
every considered-but-excluded variable records why it is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvironmentVariable:
    name: str
    value: str
    scope: str  # "global_launch" | "per_rank" | "optional_diagnostic"
    source: str
    justification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "scope": self.scope,
            "source": self.source,
            "justification": self.justification,
        }


@dataclass(frozen=True)
class EnvironmentMaterialization:
    included: tuple[EnvironmentVariable, ...]
    excluded: dict[str, str]  # variable name -> reason not included

    def as_flat_map(self) -> dict[str, str]:
        return {v.name: v.value for v in self.included}

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_launch_environment": [
                v.to_dict() for v in self.included if v.scope == "global_launch"
            ],
            "per_rank_environment": [
                v.to_dict() for v in self.included if v.scope == "per_rank"
            ],
            "optional_diagnostic_environment": [
                v.to_dict() for v in self.included if v.scope == "optional_diagnostic"
            ],
            "excluded_variables": self.excluded,
        }


def materialize_environment(
    *,
    visible_physical_devices: tuple[int, ...],
    master_address: str,
    master_port: int,
    world_size: int,
    distributed_executor_backend: str,
) -> EnvironmentMaterialization:
    included: list[EnvironmentVariable] = []
    excluded: dict[str, str] = {}

    included.append(
        EnvironmentVariable(
            name="CUDA_VISIBLE_DEVICES",
            value=",".join(str(i) for i in visible_physical_devices),
            scope="global_launch",
            source="runtime_discovery",
            justification=(
                "Restricts the launch to the exact physical GPU indices the D3B rank "
                "placement contract resolved to, preventing vLLM from opportunistically "
                "using a GPU that was not part of the validated placement."
            ),
        )
    )
    included.append(
        EnvironmentVariable(
            name="MASTER_ADDR",
            value=master_address,
            scope="global_launch",
            source="capability_profile",
            justification=(
                "vLLM's distributed process group rendezvous address; also exposed as "
                "the --master-addr CLI flag (registry default matches this value)."
            ),
        )
    )
    included.append(
        EnvironmentVariable(
            name="MASTER_PORT",
            value=str(master_port),
            scope="global_launch",
            source="capability_profile",
            justification="Rendezvous port paired with MASTER_ADDR; mirrors --master-port.",
        )
    )
    included.append(
        EnvironmentVariable(
            name="WORLD_SIZE",
            value=str(world_size),
            scope="global_launch",
            source="compiler_plan",
            justification=(
                "Recorded for provenance/documentation of expected process count; vLLM's "
                "own executor derives its internal world size from --tensor-parallel-size x "
                "--pipeline-parallel-size, so this is descriptive, not a value vLLM reads."
            ),
        )
    )
    included.append(
        EnvironmentVariable(
            name="VLLM_WORKER_MULTIPROC_METHOD",
            value="spawn",
            scope="global_launch",
            source="explicit_D3B_default",
            justification=(
                "Forces the 'mp' distributed_executor_backend to use spawn rather than "
                "fork, avoiding CUDA-context-in-fork hazards; matches the D1/D3A precedent "
                "of using multiprocessing 'spawn' for real OS-process IPC."
            ),
        )
    )
    included.append(
        EnvironmentVariable(
            name="TOKENIZERS_PARALLELISM",
            value="false",
            scope="global_launch",
            source="explicit_D3B_default",
            justification=(
                "Suppresses HuggingFace tokenizers' background-thread parallelism warning "
                "under multiprocess launch; a launch-hygiene default, not a distributed "
                "correctness requirement."
            ),
        )
    )

    included.append(
        EnvironmentVariable(
            name="NCCL_DEBUG",
            value="INFO",
            scope="optional_diagnostic",
            source="explicit_D3B_default",
            justification=(
                "Diagnostic only: would surface NCCL initialization detail if a real "
                "multi-GPU launch were ever attempted. Not required for materialization, "
                "spec generation, or dry-run validation, and has no effect since no NCCL "
                "process group is ever started by D3B."
            ),
        )
    )

    # Explicitly-considered-and-excluded variables.
    excluded["RANK"] = (
        "vLLM's OpenAI server entry point does not require the caller to set RANK; its "
        "internal 'mp'/'ray' executor assigns per-worker rank itself. Fabricating a "
        "single top-level RANK value would misrepresent a per-rank-managed variable as "
        "caller-supplied."
    )
    excluded["LOCAL_RANK"] = "Same reasoning as RANK: internally managed by vLLM's executor, not caller-supplied."
    excluded["NCCL_SOCKET_IFNAME"] = (
        "Only meaningful when multiple physical NICs must be disambiguated for "
        "multi-node NCCL rendezvous; out of scope for a single-node launch spec and not "
        "needed by any preflight or dry-run check performed here."
    )
    excluded["HF_HOME"] = (
        "Not set explicitly: the host's existing HuggingFace cache "
        "(~/.cache/huggingface) already resolves the model locally (verified during "
        "preflight's model-resolvable check); overriding it here would risk pointing "
        "away from the already-verified cache location."
    )

    return EnvironmentMaterialization(included=tuple(included), excluded=excluded)

"""D1: Compiler-Planned TP=2 Multi-Process Simulation.

Real local OS processes (multiprocessing "spawn"), real localhost IPC
(multiprocessing.Queue), one simulated collective (all_reduce sum).

Truth boundary: single-host CPU multi-process simulation, localhost IPC
only. Not NCCL, not GPU-to-GPU communication, not real vLLM tensor
parallelism, not representative of multi-GPU scaling.
"""

from deployment.tp_process_runtime.collective import CollectiveCoordinator, CollectiveOutcome
from deployment.tp_process_runtime.cross_layer_provenance import (
    CrossLayerProvenanceReport,
    verify_cross_layer_provenance,
)
from deployment.tp_process_runtime.qwen_workload import (
    QwenDerivedWorkload,
    build_qwen_derived_workload,
)
from deployment.tp_process_runtime.rank_worker import RankProcessSpec
from deployment.tp_process_runtime.reference import serial_matmul_reference
from deployment.tp_process_runtime.runtime import (
    DistributedExecutionResult,
    DistributedExecutionTrace,
    DistributedProcessRuntime,
    DistributedRuntimeError,
    ProcessRecord,
    RankMailbox,
)

__all__ = [
    "CollectiveCoordinator",
    "CollectiveOutcome",
    "CrossLayerProvenanceReport",
    "DistributedExecutionResult",
    "DistributedExecutionTrace",
    "DistributedProcessRuntime",
    "DistributedRuntimeError",
    "ProcessRecord",
    "QwenDerivedWorkload",
    "RankMailbox",
    "RankProcessSpec",
    "build_qwen_derived_workload",
    "serial_matmul_reference",
    "verify_cross_layer_provenance",
]

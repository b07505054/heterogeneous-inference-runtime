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
from deployment.tp_process_runtime.linear_tp_decomposition import (
    RankShard,
    TPDecompositionError,
    apply_bias_contract,
    apply_bias_twice_incorrectly,
    build_rank_shards,
    rank_local_partial_output,
)
from deployment.tp_process_runtime.live_qwen_provenance import (
    REQUIRED_COUNTERS,
    LiveQwenProvenanceReport,
    verify_live_qwen_provenance,
)
from deployment.tp_process_runtime.qwen_module_mapping import (
    OperatorMappingError,
    OperatorMappingResult,
    map_compiler_operator_to_module,
    parse_operator_id,
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
from deployment.tp_process_runtime.serialized_collective_replay import run_serialized_all_reduce

__all__ = [
    "REQUIRED_COUNTERS",
    "CollectiveCoordinator",
    "CollectiveOutcome",
    "CrossLayerProvenanceReport",
    "DistributedExecutionResult",
    "DistributedExecutionTrace",
    "DistributedProcessRuntime",
    "DistributedRuntimeError",
    "LiveQwenProvenanceReport",
    "OperatorMappingError",
    "OperatorMappingResult",
    "ProcessRecord",
    "QwenDerivedWorkload",
    "RankMailbox",
    "RankProcessSpec",
    "RankShard",
    "TPDecompositionError",
    "apply_bias_contract",
    "apply_bias_twice_incorrectly",
    "build_qwen_derived_workload",
    "build_rank_shards",
    "map_compiler_operator_to_module",
    "parse_operator_id",
    "rank_local_partial_output",
    "run_serialized_all_reduce",
    "serial_matmul_reference",
    "verify_cross_layer_provenance",
    "verify_live_qwen_provenance",
]

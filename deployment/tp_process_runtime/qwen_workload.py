"""D2: derive a deterministic Qwen-shaped matmul workload from a real
compiler-planned DistributedPlan.

The selected D2 operator (llm.o_proj) is a square hidden_size x hidden_size
projection: activation (sequence_length, hidden_size) times weight
(hidden_size, hidden_size), partitioned along the hidden (K) axis exactly as
declared by the compiler's tensor_shards. sequence_length is not part of the
static compiler plan (real serving sequence length is inherently dynamic),
so the runtime picks a fixed, deterministic, clearly-labeled value here --
this is not a live Transformers/vLLM tensor, only a Qwen-shaped synthetic
workload matching the selected operator's real static dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deployment.execution_plan.schema import DistributedPlan

# Runtime-chosen deterministic sequence length. Not derived from the plan
# (sequence length is dynamic in real serving); documented explicitly so it
# is never mistaken for a compiler-declared dimension.
DEFAULT_SEQUENCE_LENGTH = 8


@dataclass(frozen=True)
class QwenDerivedWorkload:
    operator_id: str
    hidden_dim: int
    sequence_length: int
    a: np.ndarray
    b: np.ndarray


def build_qwen_derived_workload(
    plan: DistributedPlan, *, seed: int = 2026,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> QwenDerivedWorkload:
    if not plan.tensor_shards:
        raise ValueError("distributed plan has no tensor_shards; cannot derive a Qwen workload")
    operator_id = plan.tensor_shards[0].tensor_id
    hidden_dim = max(s.range_end for s in plan.tensor_shards)
    if hidden_dim <= 0:
        raise ValueError("distributed plan declares a non-positive hidden dimension")

    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(sequence_length, hidden_dim))
    b = rng.uniform(-1.0, 1.0, size=(hidden_dim, hidden_dim))
    return QwenDerivedWorkload(
        operator_id=operator_id, hidden_dim=hidden_dim,
        sequence_length=sequence_length, a=a, b=b,
    )

"""D5: real workload matrix for TP=1 vs TP=2 decision-boundary discovery.

The grid, the calibration/held-out split, and the workload weighting scheme
are all declared here, once, before any measurement is taken. Nothing in
this module is allowed to change after results are observed -- the split
and weights are a fixed function of (input_length, output_length,
concurrency) alone, never of measured outcomes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

INPUT_LENGTHS = (32, 256, 1024)
OUTPUT_LENGTHS = (32, 128, 256)
CONCURRENCY_LEVELS = (1, 2, 4, 8)

# A fixed base corpus used to build real, varied English-text prompts of an
# exact target token length (via truncation against the real tokenizer) --
# never a degenerate single repeated token, which could trigger unusual
# prefix-caching/dedup behavior unrepresentative of real workloads.
BASE_CORPUS_PARAGRAPHS = (
    "The history of distributed computing systems traces back to early "
    "networked mainframes, where researchers first grappled with the "
    "fundamental tradeoffs between computation locality and communication "
    "overhead. As workloads grew, engineers discovered that naively "
    "parallelizing a task across multiple processors often introduced more "
    "synchronization cost than it saved in raw compute time, a lesson that "
    "remains directly relevant to modern GPU-based tensor parallelism.",
    "Large language model inference presents a distinctive scheduling "
    "challenge because a single request's cost is not fixed: the prefill "
    "phase processes the entire prompt in parallel, while the decode phase "
    "generates one token at a time in a tight, latency-sensitive loop. This "
    "asymmetry means that a serving system optimized purely for throughput "
    "during prefill can behave very differently once autoregressive decode "
    "dominates the request's remaining lifetime.",
    "Network interconnect topology matters enormously for tensor-parallel "
    "inference because every partitioned linear layer requires a collective "
    "communication step to reconstruct its output before the next stage of "
    "computation can proceed. On hardware without a dedicated high-bandwidth "
    "GPU-to-GPU link, this collective must traverse the same PCIe fabric "
    "used for host-to-device transfers, which can materially change the "
    "point at which additional parallelism stops paying for itself.",
    "A compiler that reasons about deployment strategy must ultimately "
    "commit to a decision before a single token has been generated, using "
    "only information that is knowable in advance: the size of the model, "
    "the shape of the anticipated workload, the memory budget of the target "
    "device, and whatever calibration measurements have already been "
    "collected from prior, comparable runs on the same hardware.",
)


def build_prompt_of_token_length(tokenizer, target_tokens: int, *, seed_offset: int = 0) -> str:
    """Builds a real, varied prompt truncated to an exact token count using
    the real tokenizer -- never a synthetic degenerate repeated token."""
    paragraphs = list(BASE_CORPUS_PARAGRAPHS)
    text = " ".join(paragraphs[(seed_offset + i) % len(paragraphs)] for i in range(target_tokens // 20 + 4))
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_tokens:
        text = text + " " + paragraphs[seed_offset % len(paragraphs)]
        ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids)


@dataclass(frozen=True)
class WorkloadSpec:
    input_length: int
    output_length: int
    concurrency: int

    @property
    def workload_id(self) -> str:
        return f"in{self.input_length}_out{self.output_length}_c{self.concurrency}"

    def to_dict(self) -> dict[str, Any]:
        return {"workload_id": self.workload_id, "input_length": self.input_length,
                "output_length": self.output_length, "concurrency": self.concurrency}


def build_full_matrix() -> list[WorkloadSpec]:
    return [
        WorkloadSpec(i, o, c)
        for i in INPUT_LENGTHS
        for o in OUTPUT_LENGTHS
        for c in CONCURRENCY_LEVELS
    ]


# D5 model-axis expansion (7B): a smaller, declared-upfront representative
# grid (short/long prompt x short/long output x concurrency 1/2/4) instead
# of the full 36-cell 0.5B grid, so more repetitions per cell are
# affordable within the same real-hardware time budget. Declared before any
# 7B measurement was taken, same as the 0.5B matrix above.
INPUT_LENGTHS_7B = (32, 1024)
OUTPUT_LENGTHS_7B = (32, 256)
CONCURRENCY_LEVELS_7B = (1, 2, 4)


def build_representative_matrix_7b() -> list[WorkloadSpec]:
    return [
        WorkloadSpec(i, o, c)
        for i in INPUT_LENGTHS_7B
        for o in OUTPUT_LENGTHS_7B
        for c in CONCURRENCY_LEVELS_7B
    ]


# ---------------------------------------------------------------------------
# Calibration / held-out split -- declared ONCE, before any measurement.
#
# Rule: a workload is held-out iff sha256(workload_id) is even-valued in its
# first byte. This is a deterministic, workload-identity-only function (no
# dependency on measured results) that yields an ~50/50 split without any
# manual per-cell curation that could be (even unconsciously) biased toward
# a desired outcome. The rule itself, and the resulting split, are written
# to calibration_holdout_split.json before any benchmark is run.
# ---------------------------------------------------------------------------

def is_held_out(spec: WorkloadSpec) -> bool:
    digest = hashlib.sha256(spec.workload_id.encode()).digest()
    return digest[0] % 2 == 0


def split_matrix(matrix: list[WorkloadSpec]) -> tuple[list[WorkloadSpec], list[WorkloadSpec]]:
    calibration = [w for w in matrix if not is_held_out(w)]
    held_out = [w for w in matrix if is_held_out(w)]
    return calibration, held_out


# ---------------------------------------------------------------------------
# Workload weighting scheme -- declared ONCE, uniform by design.
#
# A non-uniform, "realistic traffic mix" weighting would require an actual
# production traffic trace this project does not have; inventing one after
# the fact would be exactly the kind of post-hoc tuning the spec forbids.
# Uniform weighting is the honest, defensible default: every workload cell
# in the (declared) matrix counts equally toward every aggregate metric.
# ---------------------------------------------------------------------------

def workload_weight(spec: WorkloadSpec) -> float:
    return 1.0


def matrix_manifest() -> dict[str, Any]:
    matrix = build_full_matrix()
    calibration, held_out = split_matrix(matrix)
    return {
        "input_lengths": list(INPUT_LENGTHS), "output_lengths": list(OUTPUT_LENGTHS),
        "concurrency_levels": list(CONCURRENCY_LEVELS),
        "total_workloads": len(matrix),
        "calibration_workloads": [w.to_dict() for w in calibration],
        "held_out_workloads": [w.to_dict() for w in held_out],
        "calibration_count": len(calibration), "held_out_count": len(held_out),
        "split_rule": "held_out iff sha256(workload_id)[0] % 2 == 0 -- deterministic function of "
                     "workload identity only, declared before any measurement, never a function of "
                     "measured results",
        "weighting_scheme": "uniform (weight=1.0 per workload) -- declared before any measurement; "
                           "no production traffic trace was available to justify a non-uniform mix, "
                           "and inventing one post-hoc would risk fitting the weighting to a desired "
                           "conclusion",
    }

"""Candidate prompt/decode interaction term (E2E-3).

This module defines the structure only. Whether it is actually wired into
phase_model predictions is decided AFTER the controlled experiment, per the
E2E-3 slice rules ("do not choose a structure until measurements distinguish
among them", "at most one narrowly justified interaction term"). See the
E2E-3 analysis report for the calibration decision and its justification.

Structure #1 from the slice spec (binary interaction penalty) is implemented
here because it is the simplest structure and, per source-level inspection of
vllm/v1/core/sched/scheduler.py, the most directly evidenced by the scheduler
mechanism actually present: running (decode) requests are scheduled first
each step and are not preempted by new prefill work; a newly-admitted
request's full (or chunked) prefill token count is co-scheduled in the SAME
step as running decode requests, so the step's wall-clock cost is driven by
whichever admitted work dominates compute -- independent, in principle, of
the enable_chunked_prefill flag once the prefill fits within one step's token
budget (true for our 128-token prompts against max_num_batched_tokens=2048).
"""
from __future__ import annotations

from dataclasses import dataclass


def decode_interference_ms(
    *, chunked_prefill_enabled: bool, active_decode_sequences: int, admitted_prefill_tokens: int,
    calibrated_cost_per_prefill_token_ms: float,
) -> float:
    """Binary interaction penalty (slice structure #1).

    NOTE: per source inspection, this term's dependence on
    `chunked_prefill_enabled` is a HYPOTHESIS to be tested, not a fact --
    the scheduler code shows the token-budget-sharing mechanism applies
    whenever prefill and decode requests are co-scheduled in the same step,
    which can happen regardless of the chunked-prefill flag when the prompt
    fits in one step's budget. If the controlled experiment shows the effect
    is present in BOTH states, this function's chunked_prefill gate should
    be removed (see H1/H2/H6 evaluation) -- this is intentionally left as
    a parameter, not hardcoded, so the analysis script can test both forms.
    """
    if active_decode_sequences <= 0 or admitted_prefill_tokens <= 0:
        return 0.0
    if chunked_prefill_enabled:
        return 0.0
    return admitted_prefill_tokens * calibrated_cost_per_prefill_token_ms


def decode_interference_ms_chunked_prefill_independent(
    *, active_decode_sequences: int, admitted_prefill_tokens: int, calibrated_cost_per_prefill_token_ms: float,
) -> float:
    """Alternative form with the chunked_prefill gate removed -- the
    co-scheduling-cost hypothesis. Only meaningful to prefer over
    `decode_interference_ms` if measurements show comparable degradation in
    both chunked-prefill states (see H1/H2)."""
    if active_decode_sequences <= 0 or admitted_prefill_tokens <= 0:
        return 0.0
    return admitted_prefill_tokens * calibrated_cost_per_prefill_token_ms


@dataclass(frozen=True)
class InterferenceCalibration:
    calibrated_cost_per_prefill_token_ms: float | None
    form: str  # "unavailable" | "chunked_prefill_gated" | "chunked_prefill_independent"
    calibrated_from: dict | None

    def to_dict(self) -> dict:
        return {"calibrated_cost_per_prefill_token_ms": self.calibrated_cost_per_prefill_token_ms,
                "form": self.form, "calibrated_from": self.calibrated_from}


UNCALIBRATED_INTERFERENCE = InterferenceCalibration(None, "unavailable", None)

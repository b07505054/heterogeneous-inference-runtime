"""RuntimeProfileTrace schema and builder.

Converts ExecutionTraceRecorder output into a typed, immutable trace artifact
suitable for offline analysis and playback (e.g., by a PocketChef PlaybackEngine).

Not a live runtime output. Not a JSON serialiser (write_json is a convenience
export only). Not connected to ExecutionEngine directly — callers must pass an
already-populated ExecutionTraceRecorder.

Truth boundary on all produced artifacts:
  "offline_runtime_simulation_not_iphone_execution"

Data flow:
  ExecutionTraceRecorder
    → RuntimeProfileTraceBuilder.from_recorder()  → TraceVariant
    → RuntimeProfileTraceBuilder.build_trace()    → RuntimeProfileTrace

Compiler plan provenance:
  compiler_plan_source = "compiler_artifact" when the optimized variant was
    built from a real compiler ServingExecutionPlan.
  compiler_plan_source = "built_in_fixture" when the compiler artifact was
    absent and the generator used its built-in development fixture.

  do_not_use_for_demo is set to True when compiler_plan_source is
    "built_in_fixture". PocketChef must render a visible warning when
    this flag is true:
      "⚠ Development Fixture — compiler pipeline is bypassed."
    Do not implement PocketChef behavior here — this is a contract note only.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deployment.execution_trace_recorder import ExecutionTraceRecorder, TraceEvent

_DEFAULT_TRUTH_BOUNDARY = "offline_runtime_simulation_not_iphone_execution"

# Allowed values for compiler_plan_source.
COMPILER_PLAN_SOURCE_ARTIFACT = "compiler_artifact"
COMPILER_PLAN_SOURCE_FIXTURE = "built_in_fixture"


# ---------------------------------------------------------------------------
# RuntimeTimeSeries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeTimeSeries:
    """Point-in-time snapshots from a recorder run.

    Lists are parallel: index i corresponds to a single snapshot.
    Not interpolated — the PocketChef PlaybackEngine interpolates on playback.
    """

    timestamps_ms: list[float]
    queue_depth: list[int]
    memory_mb: list[float]
    active_requests: list[int]

    def to_dict(self) -> dict:
        return {
            "timestamps_ms": list(self.timestamps_ms),
            "queue_depth": list(self.queue_depth),
            "memory_mb": list(self.memory_mb),
            "active_requests": list(self.active_requests),
        }


# ---------------------------------------------------------------------------
# VariantSummary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantSummary:
    """Aggregate statistics for one trace variant.

    All latency/memory values are estimated from the simulated recorder run,
    not from measured device execution.
    """

    p50_latency_ms: float
    p95_latency_ms: float
    peak_memory_mb: float
    avg_queue_depth: float
    total_events: int
    total_requests: int
    truth_boundary: str

    def to_dict(self) -> dict:
        return {
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_queue_depth": self.avg_queue_depth,
            "total_events": self.total_events,
            "total_requests": self.total_requests,
            "truth_boundary": self.truth_boundary,
        }


# ---------------------------------------------------------------------------
# TraceVariant
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceVariant:
    """Trace output for one runtime configuration (e.g. baseline or optimized).

    events is the raw TraceEvent list from the recorder — preserved verbatim,
    no filtering or rewriting. Callers that need to compare or display specific
    events should filter from this list.
    """

    variant_id: str
    runtime_mode: str
    optimizer_features: list[str]
    total_duration_ms: float
    events: list[TraceEvent]
    timeseries: RuntimeTimeSeries
    summary: VariantSummary

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "runtime_mode": self.runtime_mode,
            "optimizer_features": list(self.optimizer_features),
            "total_duration_ms": self.total_duration_ms,
            "events": [
                {
                    "category": ev.category,
                    "name": ev.name,
                    "lane": ev.lane,
                    "start_ms": ev.start_ms,
                    "end_ms": ev.end_ms,
                    "duration_ms": ev.duration_ms,
                    "request_id": ev.request_id,
                    "metadata": dict(ev.metadata),
                    "truth_boundary": ev.truth_boundary,
                }
                for ev in self.events
            ],
            "timeseries": self.timeseries.to_dict(),
            "summary": self.summary.to_dict(),
        }


# ---------------------------------------------------------------------------
# ComparisonSummary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparisonSummary:
    """Relative deltas between baseline and optimized variants.

    All delta values are percentages. Negative = optimized is better for latency.
    Comparison is only meaningful when both variants have the same workload.
    headline is a human-readable one-liner derived from latency_delta_pct.

    When built from a fixture (compiler artifact absent), headline is prefixed
    with "[FIXTURE] Compiler artifact missing. " to make the provenance gap
    visible in any consumer that displays the headline.
    """

    latency_delta_pct: float
    peak_memory_delta_pct: float
    queue_depth_delta_pct: float
    headline: str

    def to_dict(self) -> dict:
        return {
            "latency_delta_pct": self.latency_delta_pct,
            "peak_memory_delta_pct": self.peak_memory_delta_pct,
            "queue_depth_delta_pct": self.queue_depth_delta_pct,
            "headline": self.headline,
        }


# ---------------------------------------------------------------------------
# RuntimeProfileTrace
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeProfileTrace:
    """Immutable offline trace artifact combining two runtime variants.

    Consumed by RuntimePlaybackEngine (Commit D+). Not consumed directly
    by SwiftUI. Not a live device measurement.

    Provenance fields:
      compiler_plan_source: "compiler_artifact" | "built_in_fixture"
      compiler_plan_path:   path or "fixture:built_in" when no artifact
      do_not_use_for_demo:  True when compiler_plan_source is "built_in_fixture"
      provenance_notes:     additional flags; includes "compiler_not_in_pipeline"
                            when fixture was used

    PocketChef must render a visible warning when do_not_use_for_demo is True:
      "⚠ Development Fixture — compiler pipeline is bypassed."
    (Implementation is in PocketChef, not here.)
    """

    schema_version: str = "1"
    artifact_type: str = "runtime_profile_trace"
    target_profile_id: str = ""
    model_name: str = ""
    trace_truth_boundary: str = _DEFAULT_TRUTH_BOUNDARY
    compiler_plan_ref: str = ""
    compiler_plan_source: str = COMPILER_PLAN_SOURCE_ARTIFACT
    compiler_plan_path: str = ""
    do_not_use_for_demo: bool = False
    provenance_notes: list[str] = field(default_factory=list)
    variants: dict[str, TraceVariant] = field(default_factory=dict)
    comparison_summary: ComparisonSummary = field(
        default_factory=lambda: ComparisonSummary(
            latency_delta_pct=0.0,
            peak_memory_delta_pct=0.0,
            queue_depth_delta_pct=0.0,
            headline="",
        )
    )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "target_profile_id": self.target_profile_id,
            "model_name": self.model_name,
            "trace_truth_boundary": self.trace_truth_boundary,
            "compiler_plan_ref": self.compiler_plan_ref,
            "compiler_plan_source": self.compiler_plan_source,
            "compiler_plan_path": self.compiler_plan_path,
            "do_not_use_for_demo": self.do_not_use_for_demo,
            "provenance_notes": list(self.provenance_notes),
            "variants": {k: v.to_dict() for k, v in self.variants.items()},
            "comparison_summary": self.comparison_summary.to_dict(),
        }

    def write_json(self, path: str | Path) -> None:
        """Serialize to JSON at path (creates parent directories if needed)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------

def _percentile(samples: list[float], pct: float) -> float:
    """Return the pct-th percentile of samples (0–100).

    Uses the nearest-rank method. Returns 0.0 for empty input.
    """
    if not samples:
        return 0.0
    sorted_samples = sorted(samples)
    k = max(0, int(len(sorted_samples) * pct / 100) - 1)
    k = min(k, len(sorted_samples) - 1)
    return sorted_samples[k]


def _safe_delta_pct(optimized: float, baseline: float) -> float:
    """Return (optimized - baseline) / baseline * 100, or 0.0 if baseline == 0."""
    if baseline == 0.0:
        return 0.0
    return (optimized - baseline) / baseline * 100.0


# ---------------------------------------------------------------------------
# RuntimeProfileTraceBuilder
# ---------------------------------------------------------------------------

class RuntimeProfileTraceBuilder:
    """Converts recorder output into typed trace artifact structures.

    Stateless — all methods are static. Does not run ExecutionEngine.
    Does not modify the recorder. Does not emit JSON directly (call
    RuntimeProfileTrace.write_json() for that).
    """

    @staticmethod
    def from_recorder(
        recorder: ExecutionTraceRecorder,
        *,
        variant_id: str,
        runtime_mode: str,
        optimizer_features: list[str],
        truth_boundary: str = _DEFAULT_TRUTH_BOUNDARY,
    ) -> TraceVariant:
        """Build a TraceVariant from a populated ExecutionTraceRecorder.

        Reads events(), snapshots(), latency_samples(), and current_time_ms()
        from recorder. Does not modify recorder state.
        """
        events = recorder.events()
        snapshots = recorder.snapshots()
        latency_samples = recorder.latency_samples()
        total_duration_ms = recorder.current_time_ms()

        # ── Timeseries ────────────────────────────────────────────────────
        timeseries = RuntimeTimeSeries(
            timestamps_ms=[s["time_ms"] for s in snapshots],
            queue_depth=[s["queue_depth"] for s in snapshots],
            memory_mb=[s["memory_mb"] for s in snapshots],
            active_requests=[s["active_requests"] for s in snapshots],
        )

        # ── Summary ───────────────────────────────────────────────────────
        p50 = _percentile(latency_samples, 50)
        p95 = _percentile(latency_samples, 95)

        memory_values = [s["memory_mb"] for s in snapshots]
        peak_memory_mb = max(memory_values) if memory_values else 0.0

        queue_values = [s["queue_depth"] for s in snapshots]
        avg_queue_depth = (
            sum(queue_values) / len(queue_values) if queue_values else 0.0
        )

        summary = VariantSummary(
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            peak_memory_mb=peak_memory_mb,
            avg_queue_depth=avg_queue_depth,
            total_events=len(events),
            total_requests=len(latency_samples),
            truth_boundary=truth_boundary,
        )

        return TraceVariant(
            variant_id=variant_id,
            runtime_mode=runtime_mode,
            optimizer_features=list(optimizer_features),
            total_duration_ms=total_duration_ms,
            events=list(events),
            timeseries=timeseries,
            summary=summary,
        )

    @staticmethod
    def build_comparison_summary(
        baseline: TraceVariant,
        optimized: TraceVariant,
        *,
        fixture_mode: bool = False,
    ) -> ComparisonSummary:
        """Compute relative deltas between baseline and optimized variants.

        latency_delta_pct < 0 means optimized is faster.
        When fixture_mode is True, headline is prefixed with
        "[FIXTURE] Compiler artifact missing. " to surface the provenance gap.
        """
        latency_delta_pct = _safe_delta_pct(
            optimized.summary.p95_latency_ms,
            baseline.summary.p95_latency_ms,
        )
        memory_delta_pct = _safe_delta_pct(
            optimized.summary.peak_memory_mb,
            baseline.summary.peak_memory_mb,
        )
        queue_delta_pct = _safe_delta_pct(
            optimized.summary.avg_queue_depth,
            baseline.summary.avg_queue_depth,
        )

        if latency_delta_pct < 0:
            reduction_pct = abs(latency_delta_pct)
            core_headline = (
                f"Compiler-guided runtime reduces p95 latency by {reduction_pct:.1f}%"
            )
        elif latency_delta_pct > 0:
            core_headline = (
                f"Compiler-guided runtime increases p95 latency by {latency_delta_pct:.1f}%"
            )
        else:
            core_headline = "Compiler-guided runtime shows no p95 latency change"

        headline = (
            f"[FIXTURE] Compiler artifact missing. {core_headline}"
            if fixture_mode
            else core_headline
        )

        return ComparisonSummary(
            latency_delta_pct=latency_delta_pct,
            peak_memory_delta_pct=memory_delta_pct,
            queue_depth_delta_pct=queue_delta_pct,
            headline=headline,
        )

    @staticmethod
    def build_trace(
        *,
        target_profile_id: str,
        model_name: str,
        compiler_plan_ref: str,
        baseline: TraceVariant,
        optimized: TraceVariant,
        compiler_plan_source: str = COMPILER_PLAN_SOURCE_ARTIFACT,
        compiler_plan_path: str = "",
        do_not_use_for_demo: bool = False,
        provenance_notes: list[str] | None = None,
    ) -> RuntimeProfileTrace:
        """Assemble a complete RuntimeProfileTrace from two variants."""
        fixture_mode = (compiler_plan_source == COMPILER_PLAN_SOURCE_FIXTURE)
        comparison = RuntimeProfileTraceBuilder.build_comparison_summary(
            baseline, optimized, fixture_mode=fixture_mode
        )
        return RuntimeProfileTrace(
            target_profile_id=target_profile_id,
            model_name=model_name,
            compiler_plan_ref=compiler_plan_ref,
            compiler_plan_source=compiler_plan_source,
            compiler_plan_path=compiler_plan_path,
            do_not_use_for_demo=do_not_use_for_demo,
            provenance_notes=list(provenance_notes or []),
            variants={
                baseline.variant_id: baseline,
                optimized.variant_id: optimized,
            },
            comparison_summary=comparison,
        )

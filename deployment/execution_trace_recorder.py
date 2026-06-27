"""ExecutionTraceRecorder: runtime observer for ExecutionEngine.

Accumulates trace events as a side effect of execution. Not a serialiser.
Not a RuntimeProfileTrace. No JSON. No file I/O.

ExecutionEngine calls recorder methods between and around decision stages.
The recorder owns the simulated clock, the active-stage stack, and all
accumulated data. ExecutionEngine never sets timestamps directly.

Usage:
    recorder = ExecutionTraceRecorder()
    result = engine.execute(plan, recorder=recorder)
    events = recorder.events()
    snapshots = recorder.snapshots()
    latency_samples = recorder.latency_samples()
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Simulated inter-stage timing gaps.
#
# These are gaps between decision pipeline stages, not stage durations.
# Compute stage duration is driven by the compiler plan's cost estimate;
# ExecutionEngine calls advance_clock(plan.scheduling_policy.compiler_cost_ms)
# between begin_stage("compute") and end_stage() for that phase.
# ---------------------------------------------------------------------------

SCHEDULING_PHASE_GAP_MS: float = 0.5
MEMORY_PHASE_GAP_MS: float = 0.3


# ---------------------------------------------------------------------------
# TraceEvent
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    """Record of one instrumented stage or instant event.

    Produced exclusively by ExecutionTraceRecorder. Never produced directly
    by ExecutionEngine. Does not carry RuntimeProfileTrace schema fields.
    """

    category: str        # "scheduler" | "memory" | "replay" | "backend" | "compute"
    name: str            # descriptive stage or event name
    lane: str            # "scheduler" | "kv_cache" | "runtime" | "cpu" | "gpu"
    start_ms: float
    end_ms: float
    duration_ms: float
    request_id: str | None
    metadata: dict[str, str]
    truth_boundary: str


# ---------------------------------------------------------------------------
# Internal pending-stage descriptor
# ---------------------------------------------------------------------------

@dataclass
class _PendingStage:
    category: str
    name: str
    lane: str
    start_ms: float
    request_id: str | None
    metadata: dict[str, str]
    truth_boundary: str


# ---------------------------------------------------------------------------
# ExecutionTraceRecorder
# ---------------------------------------------------------------------------

class ExecutionTraceRecorder:
    """Observer that accumulates trace events from ExecutionEngine execution.

    Owns:
      - current simulated clock (_current_time_ms)
      - active-stage slot (_pending; max depth 1 — nested begin raises)
      - accumulated trace events
      - memory/queue snapshots
      - per-request latency samples

    Not a serialiser. Callers build higher-level structures (RuntimeProfileTrace)
    from events(), snapshots(), and latency_samples() in a separate step.
    """

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._snapshots: list[dict] = []
        self._latency_samples: list[float] = []
        self._current_time_ms: float = 0.0
        self._pending: _PendingStage | None = None

    # ------------------------------------------------------------------
    # Stage instrumentation
    # ------------------------------------------------------------------

    def begin_stage(
        self,
        category: str,
        name: str,
        lane: str,
        request_id: str | None = None,
        metadata: dict[str, str] | None = None,
        truth_boundary: str = "",
    ) -> None:
        """Open a timed stage.

        Records start_ms = current_time_ms. Must be closed with end_stage()
        before another begin_stage() call. Raises RuntimeError if a stage is
        already open.
        """
        if self._pending is not None:
            raise RuntimeError(
                f"begin_stage('{name}') called while stage "
                f"'{self._pending.name}' is still active; call end_stage() first"
            )
        self._pending = _PendingStage(
            category=category,
            name=name,
            lane=lane,
            start_ms=self._current_time_ms,
            request_id=request_id,
            metadata=dict(metadata or {}),
            truth_boundary=truth_boundary,
        )

    def end_stage(self, metadata_update: dict[str, str] | None = None) -> None:
        """Close the currently open stage and emit a TraceEvent.

        end_ms and duration_ms are derived from current_time_ms at call time.
        metadata_update is merged into the event metadata (update semantics;
        existing keys are overwritten by update values). Raises RuntimeError
        if no stage is open.
        """
        if self._pending is None:
            raise RuntimeError(
                "end_stage() called without a matching begin_stage()"
            )
        p = self._pending
        self._pending = None
        if metadata_update:
            p.metadata.update(metadata_update)
        end_ms = self._current_time_ms
        self._events.append(TraceEvent(
            category=p.category,
            name=p.name,
            lane=p.lane,
            start_ms=p.start_ms,
            end_ms=end_ms,
            duration_ms=end_ms - p.start_ms,
            request_id=p.request_id,
            metadata=p.metadata,
            truth_boundary=p.truth_boundary,
        ))

    def instant_event(
        self,
        category: str,
        name: str,
        lane: str,
        request_id: str | None = None,
        metadata: dict[str, str] | None = None,
        truth_boundary: str = "",
    ) -> None:
        """Emit a zero-duration event at the current simulated clock time.

        Does not require or open a stage. Safe to call while no stage is open.
        """
        t = self._current_time_ms
        self._events.append(TraceEvent(
            category=category,
            name=name,
            lane=lane,
            start_ms=t,
            end_ms=t,
            duration_ms=0.0,
            request_id=request_id,
            metadata=dict(metadata or {}),
            truth_boundary=truth_boundary,
        ))

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def advance_clock(self, delta_ms: float) -> None:
        """Advance the simulated clock by delta_ms.

        May be called between stages (inter-stage gap) or between
        begin_stage() and end_stage() (intra-stage duration, e.g. compute).
        """
        self._current_time_ms += delta_ms

    def current_time_ms(self) -> float:
        """Return the current simulated clock value."""
        return self._current_time_ms

    # ------------------------------------------------------------------
    # Snapshot / telemetry recording
    # ------------------------------------------------------------------

    def record_snapshot(
        self,
        queue_depth: int,
        memory_mb: float,
        active_requests: int,
    ) -> None:
        """Record a point-in-time snapshot at the current simulated clock.

        Called by higher-level workload runners (e.g. the trace generator
        script), not by ExecutionEngine directly.
        """
        self._snapshots.append({
            "time_ms": self._current_time_ms,
            "queue_depth": queue_depth,
            "memory_mb": memory_mb,
            "active_requests": active_requests,
        })

    def record_request_latency(self, latency_ms: float) -> None:
        """Record the end-to-end latency of one completed request.

        Called by higher-level workload runners after each full
        prefill+decode sequence completes.
        """
        self._latency_samples.append(latency_ms)

    # ------------------------------------------------------------------
    # Accessors — return copies; callers cannot mutate recorder state
    # ------------------------------------------------------------------

    def events(self) -> list[TraceEvent]:
        """Return a shallow copy of the accumulated event list."""
        return list(self._events)

    def snapshots(self) -> list[dict]:
        """Return a shallow copy of the accumulated snapshot list."""
        return list(self._snapshots)

    def latency_samples(self) -> list[float]:
        """Return a shallow copy of the accumulated latency sample list."""
        return list(self._latency_samples)

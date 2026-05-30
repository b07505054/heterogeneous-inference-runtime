import json
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TraceEvent:
    frame_id: int
    stage: str
    start_time_s: float
    duration_ms: float
    thread_id: int
    metadata: dict = field(default_factory=dict)


class PipelineTracer:
    def __init__(self):
        self.events: list[TraceEvent] = []
        self.lock = threading.Lock()

    def record(
        self,
        frame_id: int,
        stage: str,
        start_time_s: float,
        end_time_s: float,
        **metadata,
    ):
        event = TraceEvent(
            frame_id=frame_id,
            stage=stage,
            start_time_s=start_time_s,
            duration_ms=(end_time_s - start_time_s) * 1000,
            thread_id=threading.get_ident(),
            metadata=metadata,
        )

        with self.lock:
            self.events.append(event)

    def export(self, output_path: str):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        trace_events = []

        for event in self.events:
            trace_events.append(
                {
                    "name": event.stage,
                    "cat": "pipeline",
                    "ph": "X",
                    "ts": event.start_time_s * 1_000_000,
                    "dur": event.duration_ms * 1_000,
                    "pid": 1,
                    "tid": event.thread_id,
                    "args": {
                        "frame_id": event.frame_id,
                        **event.metadata,
                    },
                }
            )

        output.write_text(json.dumps(trace_events), encoding="utf-8")
        print(f"Saved pipeline Chrome Trace to {output}")

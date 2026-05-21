import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RuntimeMetrics:
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=500))
    frames_seen: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def record_seen(self):
        self.frames_seen += 1

    def record_processed(self, latency_ms: float):
        self.frames_processed += 1
        self.latencies_ms.append(latency_ms)

    def record_dropped(self):
        self.frames_dropped += 1

    def fps(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.frames_processed / elapsed

    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def snapshot(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "fps": round(self.fps(), 3),
            "avg_latency_ms": round(self.avg_latency_ms(), 3),
        }
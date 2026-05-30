import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deployment.metrics import RuntimeMetrics
from deployment.monitoring_api import create_monitoring_app


class FakeBackend:
    name = "mock"
    requested_provider = None
    active_provider = "CPUExecutionProvider"
    actual_providers = ["CPUExecutionProvider"]


def main():
    metrics = RuntimeMetrics()
    metrics.record_seen()
    metrics.record_processed(2.5)
    metrics.record_dropped()

    pipeline = SimpleNamespace(
        metrics=metrics,
        backend=FakeBackend(),
        model_config={"name": "smoke_test_model"},
        stop_event=SimpleNamespace(is_set=lambda: False),
    )

    app = create_monitoring_app(pipeline)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    json_metrics = client.get("/metrics")
    assert json_metrics.status_code == 200
    assert json_metrics.json()["frames_seen"] == 1

    prometheus = client.get("/metrics/prometheus")
    assert prometheus.status_code == 200

    text = prometheus.text
    assert "edge_frames_seen_total" in text
    assert "edge_frames_processed_total" in text
    assert "edge_avg_latency_ms" in text

    print("Monitoring API smoke test passed.")
    print(text)


if __name__ == "__main__":
    main()

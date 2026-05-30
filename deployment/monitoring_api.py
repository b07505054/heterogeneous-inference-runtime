from fastapi import FastAPI, Response


def create_monitoring_app(pipeline):
    app = FastAPI(title="Edge CV Inference Monitoring API")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "pipeline_running": not pipeline.stop_event.is_set(),
        }

    @app.get("/metrics")
    def metrics():
        return pipeline.metrics.snapshot()

    @app.get("/backend")
    def backend():
        backend = pipeline.backend

        return {
            "backend": getattr(backend, "name", "unknown"),
            "requested_provider": getattr(backend, "requested_provider", None),
            "active_provider": getattr(backend, "active_provider", None),
            "session_providers": getattr(backend, "actual_providers", None),
        }

    @app.get("/model")
    def model():
        return pipeline.model_config or {}

    @app.get("/metrics/prometheus")
    def prometheus_metrics():
        snapshot = pipeline.metrics.snapshot()
        backend = pipeline.backend

        labels = {
            "backend": getattr(backend, "name", "unknown"),
            "active_provider": getattr(backend, "active_provider", "unknown"),
        }

        label_text = ",".join(
            f'{key}="{value}"'
            for key, value in labels.items()
        )

        body = "\n".join(
            [
                "# HELP edge_frames_seen_total Total frames seen by the pipeline.",
                "# TYPE edge_frames_seen_total counter",
                f"edge_frames_seen_total{{{label_text}}} {snapshot['frames_seen']}",
                "# HELP edge_frames_processed_total Total frames processed by the pipeline.",
                "# TYPE edge_frames_processed_total counter",
                f"edge_frames_processed_total{{{label_text}}} {snapshot['frames_processed']}",
                "# HELP edge_frames_dropped_total Total frames dropped by the pipeline.",
                "# TYPE edge_frames_dropped_total counter",
                f"edge_frames_dropped_total{{{label_text}}} {snapshot['frames_dropped']}",
                "# HELP edge_pipeline_fps Current processed frames per second.",
                "# TYPE edge_pipeline_fps gauge",
                f"edge_pipeline_fps{{{label_text}}} {snapshot['fps']}",
                "# HELP edge_avg_latency_ms Average inference latency in milliseconds.",
                "# TYPE edge_avg_latency_ms gauge",
                f"edge_avg_latency_ms{{{label_text}}} {snapshot['avg_latency_ms']}",
                "",
            ]
        )

        return Response(
            content=body,
            media_type="text/plain; version=0.0.4",
        )

    return app

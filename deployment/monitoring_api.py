from fastapi import FastAPI


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

    return app
import json
from pathlib import Path


def export_metrics(pipeline, output_path: str):
    backend = pipeline.backend

    payload = {
        "metrics": pipeline.metrics.snapshot(),
        "backend": {
            "name": getattr(backend, "name", "unknown"),
            "requested_provider": getattr(
                backend,
                "requested_provider",
                None,
            ),
            "active_provider": getattr(
                backend,
                "active_provider",
                None,
            ),
            "session_providers": getattr(
                backend,
                "actual_providers",
                None,
            ),
        },
    }

    output = Path(output_path)
    output.parent.mkdir(exist_ok=True)

    output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"Saved metrics to {output}")
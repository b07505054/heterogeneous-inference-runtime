import argparse
import queue
import threading
import time

from deployment.frame_source import VideoFrameSource
from deployment.inference_backend import MockCVBackend
from deployment.metrics import RuntimeMetrics
from deployment.onnx_cv_backend import ONNXRuntimeCVBackend
import uvicorn
from deployment.monitoring_api import create_monitoring_app
from deployment.export_metrics import export_metrics
from deployment.model_registry import ModelRegistry
from deployment.tracing import PipelineTracer


class AsyncVideoInferencePipeline:
    def __init__(
        self,
        source,
        backend,
        queue_size: int = 8,
        max_frames: int = 300,
        model_config=None,
        trace_output: str | None = None,
    ):
        
        self.source = VideoFrameSource(source)
        self.backend = backend
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.metrics = RuntimeMetrics()
        self.max_frames = max_frames
        self.stop_event = threading.Event()
        self.model_config = model_config
        self.trace_output = trace_output
        self.tracer = PipelineTracer() if trace_output else None

    def capture_loop(self):
        frame_id = 0

        while not self.stop_event.is_set() and frame_id < self.max_frames:
            read_start = time.perf_counter()
            frame = self.source.read()
            read_end = time.perf_counter()

            if frame is None:
                break
            if self.tracer:
                self.tracer.record(
                    frame_id,
                    "capture",
                    read_start,
                    read_end,
                )
            self.metrics.record_seen()

            item = {
                "frame_id": frame_id,
                "frame": frame,
                "timestamp": time.perf_counter(),
            }

            try:
                enqueue_start = time.perf_counter()
                self.frame_queue.put_nowait(item)
            except queue.Full:
                drop_end = time.perf_counter()
                self.metrics.record_dropped()

                if self.tracer:
                    self.tracer.record(
                        frame_id,
                        "queue_drop",
                        enqueue_start,
                        drop_end,
                        queue_size=self.frame_queue.qsize(),
                    )
            else:
                enqueue_end = time.perf_counter()

                if self.tracer:
                    self.tracer.record(
                        frame_id,
                        "queue_enqueue",
                        enqueue_start,
                        enqueue_end,
                        queue_size=self.frame_queue.qsize(),
                    )

            frame_id += 1

        self.stop_event.set()

    def inference_loop(self):
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                item = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            queue_wait_start = item["timestamp"]
            queue_wait_end = time.perf_counter()

            if self.tracer:
                self.tracer.record(
                    item["frame_id"],
                    "queue_wait",
                    queue_wait_start,
                    queue_wait_end,
                )

            start = time.perf_counter()
            result = self.backend.infer(item["frame"])
            end = time.perf_counter()

            if self.tracer:
                self.tracer.record(
                    item["frame_id"],
                    "inference",
                    start,
                    end,
                    backend=result["backend"],
                )

            latency_ms = (end - start) * 1000
            metrics_start = time.perf_counter()
            self.metrics.record_processed(latency_ms)
            metrics_end = time.perf_counter()

            if self.tracer:
                self.tracer.record(
                    item["frame_id"],
                    "metrics_update",
                    metrics_start,
                    metrics_end,
                )

            if item["frame_id"] % 30 == 0:
                print(
                    {
                        "frame_id": item["frame_id"],
                        "backend": result["backend"],
                        "latency_ms": round(latency_ms, 3),
                        "metrics": self.metrics.snapshot(),
                    }
                )

    def run(self):
        capture_thread = threading.Thread(target=self.capture_loop)
        inference_thread = threading.Thread(target=self.inference_loop)

        capture_thread.start()
        inference_thread.start()

        capture_thread.join()
        inference_thread.join()

        self.source.close()

        print("Final metrics:")
        print(self.metrics.snapshot())

        if self.tracer and self.trace_output:
            self.tracer.export(self.trace_output)


def parse_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback-provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--backend", type=str, default="mock")
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--enable-api", action="store_true")
    parser.add_argument("--api-host", type=str, default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="results/video_pipeline_metrics.json",
    )
    parser.add_argument(
        "--trace-output",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="configs/model_registry.json",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    registry = ModelRegistry(args.registry)

    model_config = (
        registry.get_model(args.model)
        if args.model is not None
        else registry.active_model_config()
    )

    print(
        {
            "model_registry": args.registry,
            "selected_model": model_config["name"],
            "backend": model_config["backend"],
            "provider": model_config.get("provider"),
            "precision": model_config.get("precision"),
        }
    )
    if args.backend == "onnx":
        backend = ONNXRuntimeCVBackend(
            model_path=model_config["model_path"],
            provider=model_config.get(
                "provider",
                args.provider,
            ),
            fallback_provider=model_config.get(
                "fallback_provider",
                args.fallback_provider,
            ),
        )
    else:
        backend = MockCVBackend()

    pipeline = AsyncVideoInferencePipeline(
        source=parse_source(args.source),
        backend=backend,
        queue_size=args.queue_size,
        max_frames=args.max_frames,
        model_config=model_config,
        trace_output=args.trace_output,
    )
    if args.enable_api:
        app = create_monitoring_app(pipeline)

        api_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={
                "app": app,
                "host": args.api_host,
                "port": args.api_port,
                "log_level": "warning",
            },
            daemon=True,
        )

        api_thread.start()

    pipeline.run()
    export_metrics(
        pipeline,
        args.metrics_output,
    )


if __name__ == "__main__":
    main()

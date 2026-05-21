import time
import numpy as np


class MockCVBackend:
    name = "MockCVBackend"

    def infer(self, frame: np.ndarray) -> dict:
        start = time.perf_counter()

        # Placeholder for real CV inference.
        # Later this will route to ONNX Runtime / TensorRT / ExecuTorch.
        height, width = frame.shape[:2]

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "backend": self.name,
            "input_shape": [height, width, 3],
            "latency_ms": latency_ms,
            "detections": [],
        }
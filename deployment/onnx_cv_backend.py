import time

import cv2
import numpy as np
import onnxruntime as ort


class ONNXRuntimeCVBackend:
    name = "ONNXRuntimeCVBackend"

    def __init__(
        self,
        model_path: str = "models/mobilenet_v2_optimized.onnx",
        provider: str = "CPUExecutionProvider",
        fallback_provider: str = "CPUExecutionProvider",
    ):
        self.model_path = model_path
        self.requested_provider = provider
        self.fallback_provider = fallback_provider

        available = ort.get_available_providers()

        if provider in available:
            if provider == fallback_provider:
                providers = [provider]
            else:
                providers = [provider, fallback_provider]
            self.active_provider = provider
        else:
            providers = [fallback_provider]
            self.active_provider = fallback_provider

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
        )

        self.actual_providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(
            {
                "backend": self.name,
                "requested_provider": self.requested_provider,
                "active_provider": self.active_provider,
                "available_providers": available,
                "session_providers": self.actual_providers,
            }
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        image = cv2.resize(frame, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return image

    def infer(self, frame: np.ndarray) -> dict:
        start = time.perf_counter()

        input_tensor = self.preprocess(frame)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )

        logits = outputs[0]
        top1 = int(np.argmax(logits, axis=1)[0])

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "backend": self.name,
            "requested_provider": self.requested_provider,
            "active_provider": self.active_provider,
            "session_providers": self.actual_providers,
            "latency_ms": latency_ms,
            "top1": top1,
        }
from __future__ import annotations

import time


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception:
        return False
    return True


class PyTorchMobileNetV2Backend:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.sample = None
        self.torch = None

    def setup(self, sample=None) -> dict:
        import torch
        from torchvision import models

        if self.device == "mps" and not torch.backends.mps.is_available():
            return {"status": "unavailable", "reason": "mps_not_available", "metrics": {}}

        torch.manual_seed(0)
        device = torch.device(self.device)
        self.model = models.mobilenet_v2(weights=None).eval().to(device)
        if sample is None:
            self.sample = torch.randn(1, 3, 224, 224, device=device)
        else:
            self.sample = torch.as_tensor(sample, dtype=torch.float32, device=device)
        load_start = time.perf_counter()
        self.model = self.model.to(device)
        load_ms = (time.perf_counter() - load_start) * 1000.0
        self.torch = torch
        return {"status": "ok", "model_load_ms": round(load_ms, 6)}

    def execute(self):
        with self.torch.no_grad():
            return self.model(self.sample)

    def sync(self) -> None:
        device_type = self.device
        if device_type == "cuda":
            self.torch.cuda.synchronize()
        elif device_type == "mps":
            self.torch.mps.synchronize()

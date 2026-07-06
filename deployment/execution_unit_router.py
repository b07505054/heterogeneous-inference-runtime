"""Map compiler execution-unit vocabulary to runtime adapter IDs.

Compiler ExecutionPlan.backend.selected_backend uses execution-unit vocabulary
(cuda_triton, cuda_cublas, cpu, coreml_ane, arm_compute). Runtime adapter IDs
are internal routing strings used by BackendAdapter implementations (vllm,
custom_cuda, pytorch_reference, coreml).

The compiler never uses runtime product names (vllm, tensorrt, etc.) as
selected_backend. This module is the single place that translates between
the two vocabularies.
"""

from __future__ import annotations


_SERVING_ADAPTER: dict[str, str] = {
    "cuda_triton":  "vllm",
    "cuda_cublas":  "vllm",
    "cpu":          "pytorch_reference",
    "coreml_ane":   "coreml",
    "arm_compute":  "coreml",
}

_KERNEL_ADAPTER: dict[str, str] = {
    "cuda_triton": "custom_cuda",
    "cuda_cublas": "custom_cuda",
}


class ExecutionUnitRouter:
    @staticmethod
    def serving_adapter_id(execution_unit: str) -> str | None:
        """Return the serving adapter ID for a compiler execution unit, or None if unknown."""
        return _SERVING_ADAPTER.get(execution_unit)

    @staticmethod
    def kernel_adapter_id(execution_unit: str) -> str | None:
        """Return the kernel adapter ID for a compiler execution unit, or None if unknown."""
        return _KERNEL_ADAPTER.get(execution_unit)

import pytest

from deployment.execution_unit_router import ExecutionUnitRouter


# ---------------------------------------------------------------------------
# serving_adapter_id
# ---------------------------------------------------------------------------

def test_cuda_triton_routes_to_vllm():
    assert ExecutionUnitRouter.serving_adapter_id("cuda_triton") == "vllm"


def test_cuda_cublas_routes_to_vllm():
    assert ExecutionUnitRouter.serving_adapter_id("cuda_cublas") == "vllm"


def test_cpu_routes_to_pytorch_reference():
    assert ExecutionUnitRouter.serving_adapter_id("cpu") == "pytorch_reference"


def test_coreml_ane_routes_to_coreml():
    assert ExecutionUnitRouter.serving_adapter_id("coreml_ane") == "coreml"


def test_arm_compute_routes_to_coreml():
    assert ExecutionUnitRouter.serving_adapter_id("arm_compute") == "coreml"


def test_unknown_execution_unit_returns_none():
    assert ExecutionUnitRouter.serving_adapter_id("vllm") is None
    assert ExecutionUnitRouter.serving_adapter_id("tensorrt") is None
    assert ExecutionUnitRouter.serving_adapter_id("") is None
    assert ExecutionUnitRouter.serving_adapter_id("unknown_unit") is None


def test_runtime_product_names_are_not_in_serving_vocab():
    # Compiler never uses these as selected_backend; they must not route silently.
    for product_name in ("vllm", "tensorrt", "onnxruntime", "pytorch"):
        assert ExecutionUnitRouter.serving_adapter_id(product_name) is None


# ---------------------------------------------------------------------------
# kernel_adapter_id
# ---------------------------------------------------------------------------

def test_cuda_triton_kernel_routes_to_custom_cuda():
    assert ExecutionUnitRouter.kernel_adapter_id("cuda_triton") == "custom_cuda"


def test_cuda_cublas_kernel_routes_to_custom_cuda():
    assert ExecutionUnitRouter.kernel_adapter_id("cuda_cublas") == "custom_cuda"


def test_cpu_kernel_returns_none():
    assert ExecutionUnitRouter.kernel_adapter_id("cpu") is None


def test_coreml_ane_kernel_returns_none():
    assert ExecutionUnitRouter.kernel_adapter_id("coreml_ane") is None


def test_unknown_kernel_unit_returns_none():
    assert ExecutionUnitRouter.kernel_adapter_id("vllm") is None
    assert ExecutionUnitRouter.kernel_adapter_id("") is None

import numpy as np
import pytest

from tvm_experiments.matmul_bias_relu import (
    MatmulBiasReluShape,
    compile_module,
    create_scheduled_module,
    create_unscheduled_module,
    import_tvm,
    make_inputs,
    numpy_reference,
    run_module,
)


pytestmark = pytest.mark.skipif(
    pytest.importorskip("importlib").util.find_spec("tvm") is None,
    reason="Apache TVM is not installed",
)


def require_tvm_llvm():
    tvm = import_tvm()
    if not tvm.runtime.enabled("llvm"):
        pytest.skip("TVM was built without LLVM support")
    return tvm


@pytest.mark.parametrize(
    "shape",
    [
        MatmulBiasReluShape(16, 16, 16),
        MatmulBiasReluShape(32, 32, 32),
    ],
)
def test_tvm_unscheduled_and_scheduled_match_numpy(shape):
    require_tvm_llvm()
    inputs = make_inputs(shape, seed=shape.m * 10000 + shape.n * 100 + shape.k)
    expected = numpy_reference(*inputs)

    unscheduled = compile_module(create_unscheduled_module(shape))
    scheduled = compile_module(create_scheduled_module(shape))

    np.testing.assert_allclose(run_module(unscheduled, shape, inputs), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(run_module(scheduled, shape, inputs), expected, rtol=1e-4, atol=1e-4)


def test_scheduled_tensorir_contains_parallel_and_vectorized_loops():
    require_tvm_llvm()
    shape = MatmulBiasReluShape(32, 32, 32)
    tensorir = create_scheduled_module(shape).script()

    assert "T.parallel" in tensorir
    assert "T.vectorized" in tensorir
    assert "matmul_bias_relu" in tensorir

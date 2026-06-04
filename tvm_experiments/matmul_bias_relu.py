"""Executable TVM TensorIR MatMul + Bias + ReLU workload.

This module intentionally keeps TVM as an optional dependency. The runtime repo
can run its normal CI without TVM, while a developer with TVM installed can
execute and benchmark the TensorIR schedule path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MatmulBiasReluShape:
    m: int
    n: int
    k: int
    dtype: str = "float32"

    @property
    def bucket(self) -> str:
        return f"{self.m}x{self.n}x{self.k}:{self.dtype}"


@dataclass(frozen=True)
class ScheduleConfig:
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 8
    vectorize_n: int = 16

    def as_dict(self) -> dict[str, int]:
        return {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "vectorize_n": self.vectorize_n,
        }


def import_tvm() -> Any:
    try:
        import tvm  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when TVM is absent.
        raise RuntimeError(
            "Apache TVM is not installed. Build TVM or set PYTHONPATH/TVM_LIBRARY_PATH "
            "before running the TensorIR benchmark."
        ) from exc
    return tvm


def build_prim_func(shape: MatmulBiasReluShape) -> Any:
    tvm = import_tvm()
    from tvm import te  # type: ignore

    m, n, k_dim = shape.m, shape.n, shape.k
    k = te.reduce_axis((0, k_dim), name="k")
    a = te.placeholder((m, k_dim), name="A", dtype=shape.dtype)
    b = te.placeholder((k_dim, n), name="B", dtype=shape.dtype)
    bias = te.placeholder((n,), name="bias", dtype=shape.dtype)

    matmul = te.compute(
        (m, n),
        lambda i, j: te.sum(a[i, k] * b[k, j], axis=k),
        name="matmul",
    )
    out = te.compute(
        (m, n),
        lambda i, j: te.max(matmul[i, j] + bias[j], tvm.tirx.const(0.0, shape.dtype)),
        name="matmul_bias_relu",
    )
    return te.create_prim_func([a, b, bias, out])


def create_unscheduled_module(shape: MatmulBiasReluShape) -> Any:
    tvm = import_tvm()
    return tvm.s_tir.Schedule(build_prim_func(shape), debug_mask="all").mod


def create_scheduled_module(
    shape: MatmulBiasReluShape,
    config: ScheduleConfig = ScheduleConfig(),
) -> Any:
    tvm = import_tvm()
    schedule = tvm.s_tir.Schedule(build_prim_func(shape), debug_mask="all")

    matmul = schedule.get_sblock("matmul")
    i, j, k = schedule.get_loops(matmul)
    i_outer, i_inner = schedule.split(i, factors=[None, config.tile_m])
    j_outer, j_inner = schedule.split(j, factors=[None, config.tile_n])
    k_outer, k_inner = schedule.split(k, factors=[None, config.tile_k])
    schedule.reorder(i_outer, j_outer, k_outer, i_inner, k_inner, j_inner)
    schedule.parallel(i_outer)
    schedule.vectorize(j_inner)

    relu = schedule.get_sblock("matmul_bias_relu")
    schedule.reverse_compute_at(relu, j_outer)

    return schedule.mod


def compile_module(module: Any, target: str = "llvm") -> Any:
    tvm = import_tvm()
    executable = tvm.compile(module, target=target)
    return executable.jit()


def make_inputs(shape: MatmulBiasReluShape, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((shape.m, shape.k)).astype(shape.dtype)
    b = rng.standard_normal((shape.k, shape.n)).astype(shape.dtype)
    bias = rng.standard_normal((shape.n,)).astype(shape.dtype)
    return a, b, bias


def numpy_reference(a: np.ndarray, b: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.maximum(a @ b + bias, 0.0).astype(a.dtype)


def run_module(runtime_module: Any, shape: MatmulBiasReluShape, inputs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    tvm = import_tvm()
    dev = tvm.cpu(0)
    a, b, bias = inputs
    out = np.zeros((shape.m, shape.n), dtype=shape.dtype)
    tvm_a = tvm.runtime.tensor(a, dev)
    tvm_b = tvm.runtime.tensor(b, dev)
    tvm_bias = tvm.runtime.tensor(bias, dev)
    tvm_out = tvm.runtime.tensor(out, dev)
    runtime_module["main"](tvm_a, tvm_b, tvm_bias, tvm_out)
    return tvm_out.numpy()


def benchmark_module(
    runtime_module: Any,
    shape: MatmulBiasReluShape,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    number: int,
    repeat: int,
) -> list[float]:
    tvm = import_tvm()
    dev = tvm.cpu(0)
    a, b, bias = inputs
    out = np.zeros((shape.m, shape.n), dtype=shape.dtype)
    tvm_args = [
        tvm.runtime.tensor(a, dev),
        tvm.runtime.tensor(b, dev),
        tvm.runtime.tensor(bias, dev),
        tvm.runtime.tensor(out, dev),
    ]
    timer = runtime_module.time_evaluator("main", dev, number=number, repeat=repeat)
    return [seconds * 1000.0 / number for seconds in timer(*tvm_args).results]

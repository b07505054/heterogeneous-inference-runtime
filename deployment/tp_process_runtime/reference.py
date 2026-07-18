"""TP1 single-rank serial reference. Computed independently of any rank
process -- never reuses a rank's partial result -- so it is a genuine
external correctness oracle for the distributed run."""

from __future__ import annotations

import numpy as np


def serial_matmul_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes for matmul: {a.shape} x {b.shape}")
    return a @ b

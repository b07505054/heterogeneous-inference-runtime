"""E2E-8: runtime helper implementing the E2E-7 GEMV-loop decomposition as a
drop-in numerical replacement for torch.nn.functional.linear.

Same contract as F.linear:
  input:  x [..., K], weight [N, K], bias [N] or None
  output: [..., N], same dtype, same device, same row order

Guarded by env vars so the default (disabled) path is byte-for-byte the
production F.linear call:
  VLLM_TINY_M_GEMV_ENABLE=1        -- master switch, default off
  VLLM_TINY_M_GEMV_THRESHOLD=8     -- max M to decompose (inclusive), default 8
  VLLM_TINY_M_GEMV_OPS=lm_head     -- comma-separated operation-family allowlist

Instrumentation (optional, off by default) records counts, not values, to
keep overhead negligible when enabled during a measurement run other than
the dedicated counter-verification pass.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val is not None else default


def _env_set(name: str, default: set[str]) -> set[str]:
    val = os.environ.get(name)
    if val is None:
        return default
    return {x.strip() for x in val.split(",") if x.strip()}


ENABLE_ENV = "VLLM_TINY_M_GEMV_ENABLE"
THRESHOLD_ENV = "VLLM_TINY_M_GEMV_THRESHOLD"
OPS_ENV = "VLLM_TINY_M_GEMV_OPS"
INSTRUMENT_ENV = "VLLM_TINY_M_GEMV_INSTRUMENT"

DEFAULT_THRESHOLD = 8
DEFAULT_OPS = {"lm_head"}


@dataclass
class DispatchCounters:
    optimized_calls: int = 0
    fallback_calls: int = 0
    observed_m: dict[int, int] = field(default_factory=dict)
    observed_nk: dict[tuple[int, int], int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, *, optimized: bool, m: int, n: int, k: int) -> None:
        with self._lock:
            if optimized:
                self.optimized_calls += 1
            else:
                self.fallback_calls += 1
            self.observed_m[m] = self.observed_m.get(m, 0) + 1
            self.observed_nk[(n, k)] = self.observed_nk.get((n, k), 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "optimized_calls": self.optimized_calls, "fallback_calls": self.fallback_calls,
                "observed_m": dict(self.observed_m),
                "observed_nk": {f"{n}x{k}": v for (n, k), v in self.observed_nk.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self.optimized_calls = 0
            self.fallback_calls = 0
            self.observed_m.clear()
            self.observed_nk.clear()


COUNTERS = DispatchCounters()


def _assembly_cat(x2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Variant 1: row list + torch.cat."""
    rows = [F.linear(x2d[i : i + 1], weight, bias) for i in range(x2d.shape[0])]
    return torch.cat(rows, dim=0)


def _assembly_preallocate(x2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Variant 2: preallocated output buffer + per-row copy_."""
    m = x2d.shape[0]
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=x2d.dtype, device=x2d.device)
    for i in range(m):
        out[i : i + 1].copy_(F.linear(x2d[i : i + 1], weight, bias))
    return out


ASSEMBLY_VARIANTS = {"cat": _assembly_cat, "preallocate": _assembly_preallocate}
_DEFAULT_ASSEMBLY = "cat"  # selected by scripts/isolated_assembly_benchmark.py measurement, see report


def tiny_m_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *,
    threshold: int = DEFAULT_THRESHOLD, assembly: str = _DEFAULT_ASSEMBLY, instrument: bool = False,
    op_family: str = "unknown",
) -> torch.Tensor:
    original_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1])
    m = x2d.shape[0]
    n, k = weight.shape[0], weight.shape[1]

    if m <= 1 or m > threshold:
        if instrument:
            COUNTERS.record(optimized=False, m=m, n=n, k=k)
        return F.linear(x, weight, bias)

    if instrument:
        COUNTERS.record(optimized=True, m=m, n=n, k=k)
    assemble = ASSEMBLY_VARIANTS[assembly]
    out2d = assemble(x2d, weight, bias)
    return out2d.reshape(*original_shape[:-1], n)


def dispatch_enabled(op_family: str) -> tuple[bool, int]:
    """Returns (enabled_for_this_family, threshold), read fresh each call so
    the flag can be toggled between baseline/optimized server launches
    without any code change -- only the env var differs."""
    enabled = _env_flag(ENABLE_ENV, False)
    ops = _env_set(OPS_ENV, DEFAULT_OPS)
    threshold = _env_int(THRESHOLD_ENV, DEFAULT_THRESHOLD)
    return (enabled and op_family in ops), threshold


def maybe_tiny_m_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *, op_family: str,
) -> torch.Tensor:
    """The actual call site helper: reads env-controlled flags fresh each
    call (cheap; a handful of os.environ.get calls) and either dispatches to
    tiny_m_linear or falls straight through to the byte-identical production
    F.linear call when disabled."""
    enabled, threshold = dispatch_enabled(op_family)
    if not enabled:
        return F.linear(x, weight, bias)
    instrument = _env_flag(INSTRUMENT_ENV, False)
    return tiny_m_linear(x, weight, bias, threshold=threshold, instrument=instrument, op_family=op_family)

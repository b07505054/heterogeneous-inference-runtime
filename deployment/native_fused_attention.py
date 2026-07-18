"""Fail-closed ctypes binding for bounded-memory native fused attention."""
from __future__ import annotations
import ctypes
import hashlib
from pathlib import Path
from typing import Any
import torch

ABI_VERSION = "hir.fused_online_attention.v1"
SYMBOLS = {
    "native_scalar": "hir_fused_online_attention_scalar",
    "native_avx2": "hir_fused_online_attention_avx2",
}

class NativeFusedAttentionError(RuntimeError): pass

class Params(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.POINTER(ctypes.c_float)), ("k", ctypes.POINTER(ctypes.c_float)),
        ("v", ctypes.POINTER(ctypes.c_float)), ("output", ctypes.POINTER(ctypes.c_float)),
        ("batch", ctypes.c_int64), ("query_length", ctypes.c_int64),
        ("context_length", ctypes.c_int64), ("query_heads", ctypes.c_int64),
        ("kv_heads", ctypes.c_int64), ("head_dimension", ctypes.c_int64),
        ("q_strides", ctypes.c_int64 * 4), ("k_strides", ctypes.c_int64 * 4),
        ("v_strides", ctypes.c_int64 * 4), ("out_strides", ctypes.c_int64 * 4),
        ("scale", ctypes.c_float), ("causal", ctypes.c_int32),
        ("query_position_offset", ctypes.c_int64), ("query_tile", ctypes.c_int64),
        ("key_tile", ctypes.c_int64), ("worker_count", ctypes.c_int64),
        ("query_head_offset", ctypes.c_int64), ("total_query_heads", ctypes.c_int64),
    ]

class Stats(ctypes.Structure):
    _fields_ = [(x, ctypes.c_uint64) for x in (
        "temporary_bytes", "largest_allocation_bytes", "allocations_per_invocation",
        "allocations_per_query_row", "score_tile_bytes", "output_accumulator_bytes",
        "worker_scratch_bytes")]

class Status(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int32), ("message", ctypes.c_char_p)]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class NativeFusedAttentionLibrary:
    def __init__(self, artifact: str | Path, expected_sha256: str, abi_version: str):
        path = Path(artifact).resolve()
        if not path.is_file(): raise NativeFusedAttentionError("native_artifact_missing")
        if sha256(path) != expected_sha256: raise NativeFusedAttentionError("native_artifact_hash_mismatch")
        if abi_version != ABI_VERSION: raise NativeFusedAttentionError("native_abi_version_mismatch")
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lib.hir_fused_attention_artifact_version.restype = ctypes.c_char_p
        if self.lib.hir_fused_attention_artifact_version().decode() != ABI_VERSION:
            raise NativeFusedAttentionError("loaded_native_abi_version_mismatch")
        self.lib.hir_fused_attention_has_avx2.restype = ctypes.c_int32

    @property
    def has_avx2(self) -> bool:
        return bool(self.lib.hir_fused_attention_has_avx2())

    def run(self, implementation: str, q: torch.Tensor, k: torch.Tensor,
            v: torch.Tensor, scale: float, query_tile: int, key_tile: int,
            *, query_head_offset: int = 0, total_query_heads: int | None = None
            ) -> tuple[torch.Tensor, dict[str, Any]]:
        if implementation not in SYMBOLS: raise NativeFusedAttentionError("unknown_native_implementation")
        if any(x.device.type != "cpu" or x.dtype != torch.float32 for x in (q,k,v)):
            raise NativeFusedAttentionError("native_requires_cpu_fp32")
        if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
            raise NativeFusedAttentionError("native_requires_rank4_compatible_qkv")
        b,h,q_len,d = q.shape; bk,kvh,context,kd = k.shape
        if b != bk or d != kd: raise NativeFusedAttentionError("native_qkv_shape_mismatch")
        if total_query_heads is None: total_query_heads = h
        output = torch.empty_like(q, memory_format=torch.contiguous_format)
        ptr = ctypes.POINTER(ctypes.c_float)
        arr = lambda xs: (ctypes.c_int64 * 4)(*map(int, xs))
        p = Params(
            ctypes.cast(q.data_ptr(), ptr), ctypes.cast(k.data_ptr(), ptr),
            ctypes.cast(v.data_ptr(), ptr), ctypes.cast(output.data_ptr(), ptr),
            b,q_len,context,h,kvh,d,arr(q.stride()),arr(k.stride()),arr(v.stride()),
            arr(output.stride()),scale,1,context-q_len,query_tile,key_tile,1,
            query_head_offset,total_query_heads)
        stats = Stats()
        fn = getattr(self.lib, SYMBOLS[implementation])
        fn.argtypes = [ctypes.POINTER(Params), ctypes.POINTER(Stats)]
        fn.restype = Status
        status = fn(ctypes.byref(p), ctypes.byref(stats))
        if status.code:
            raise NativeFusedAttentionError((status.message or b"native_error").decode())
        return output, {
            "algorithm": "fused_tiled_online_softmax",
            "implementation": implementation,
            "native_symbol": SYMBOLS[implementation],
            "full_score_materialized": False, "full_probability_materialized": False,
            "score_bytes": 0, "probability_bytes": 0,
            "temporary_bytes": stats.temporary_bytes,
            "largest_temporary_allocation_bytes": stats.largest_allocation_bytes,
            "tracked_temporary_allocations": stats.allocations_per_invocation,
            "allocations_per_query_row": stats.allocations_per_query_row,
            "score_tile_bytes": stats.score_tile_bytes,
            "output_accumulator_bytes": stats.output_accumulator_bytes,
            "worker_scratch_bytes": stats.worker_scratch_bytes,
        }

"""Analytical memory model: weight bytes and KV-cache bytes.

All quantities here are theoretical tensor/traffic byte counts derived from
model metadata. They are NOT forced to match vLLM's observed allocation --
see calibration_row.py for the explicit theoretical-vs-block-rounded-vs-
observed comparison and the reasons they diverge.
"""
from __future__ import annotations

from dataclasses import dataclass

from perf_model.schema import HardwareFeatures, ModelFeatures, MetricEstimate

_DTYPE_BYTES = {
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "float32": 4, "fp32": 4,
    "float8": 1, "fp8": 1, "int8": 1,
}


def bytes_per_element(dtype: str, quantization: str) -> int:
    if quantization and quantization.lower() not in ("none", "", "auto"):
        # Quantized weight-store width is not the same as the compute dtype;
        # v1 only handles the identity case explicitly and fails closed
        # otherwise rather than guessing a bit-width.
        raise ValueError(
            f"quantization={quantization!r} weight-byte-width not modeled in v1; "
            "extend _DTYPE_BYTES / add a quant table before using this path"
        )
    key = dtype.lower()
    if key not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype for byte-width lookup: {dtype!r}")
    return _DTYPE_BYTES[key]


def weight_memory_bytes(model: ModelFeatures) -> int:
    """weight_memory = parameter_count * bytes_per_parameter.

    No quantization/tying overhead adjustment beyond the parameter count
    already reflecting tied embeddings (ModelFeatures.parameter_count is
    read from the HF config's declared architecture, which already accounts
    for tie_word_embeddings when computed by the caller).
    """
    return model.parameter_count * bytes_per_element(model.dtype, model.quantization)


def kv_bytes_per_token(model: ModelFeatures, kv_cache_dtype_bytes: int) -> int:
    """KV_bytes_per_token = layers * 2 (K and V) * kv_heads * head_dim * bytes_per_kv_element."""
    return model.layer_count * 2 * model.kv_head_count * model.head_dimension * kv_cache_dtype_bytes


@dataclass(frozen=True)
class KVMemoryEstimate:
    theoretical_bytes: int
    block_rounded_bytes: int | None
    per_token_bytes: int
    total_live_tokens: int
    block_size_used: int | None


def kv_peak_bytes(
    model: ModelFeatures,
    *,
    kv_cache_dtype_bytes: int,
    per_sequence_token_counts: list[int],
    block_size: int | None = None,
) -> KVMemoryEstimate:
    """KV_peak = KV_bytes_per_token * total_live_tokens.

    Block-rounding is applied ONLY if a resolved block_size is supplied
    (vLLM allocates per-sequence in block-size units, so per-sequence
    token counts are rounded up to the next block boundary before summing;
    global/theoretical bytes are reported unrounded for comparison).
    """
    per_token = kv_bytes_per_token(model, kv_cache_dtype_bytes)
    theoretical_tokens = sum(per_sequence_token_counts)
    theoretical = per_token * theoretical_tokens

    block_rounded = None
    if block_size:
        rounded_tokens = sum(
            ((count + block_size - 1) // block_size) * block_size
            for count in per_sequence_token_counts
        )
        block_rounded = per_token * rounded_tokens

    return KVMemoryEstimate(
        theoretical_bytes=theoretical,
        block_rounded_bytes=block_rounded,
        per_token_bytes=per_token,
        total_live_tokens=theoretical_tokens,
        block_size_used=block_size,
    )


def total_predicted_memory_bytes(
    *,
    weight_bytes: int,
    kv_peak_estimate: KVMemoryEstimate,
    runtime_overhead_bytes: int,
    safety_margin_bytes: int,
    prefer_block_rounded: bool = True,
) -> int:
    kv_component = (
        kv_peak_estimate.block_rounded_bytes
        if prefer_block_rounded and kv_peak_estimate.block_rounded_bytes is not None
        else kv_peak_estimate.theoretical_bytes
    )
    return weight_bytes + kv_component + runtime_overhead_bytes + safety_margin_bytes


def predict_oom(total_predicted_bytes: int, hardware: HardwareFeatures) -> MetricEstimate:
    gpu_bytes = hardware.gpu_memory_bytes.value
    if gpu_bytes is None:
        return MetricEstimate(
            value=None, method="unsupported", truth_boundary="unsupported_no_estimate",
            note="hardware.gpu_memory_bytes unknown",
        )
    truth = (
        "analytical_no_measurement"
        if hardware.gpu_memory_bytes.source_class in ("measured", "device_reported")
        else "analytical_with_phase_derived_constant"
    )
    return MetricEstimate(
        value=bool(total_predicted_bytes > gpu_bytes),
        method="analytical",
        truth_boundary=truth,
        note=(
            f"predicted_total={total_predicted_bytes}B vs gpu_memory={gpu_bytes}B "
            f"({hardware.gpu_memory_bytes.source_class})"
        ),
    )

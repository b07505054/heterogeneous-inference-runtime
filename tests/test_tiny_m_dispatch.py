import os

import pytest
import torch
import torch.nn.functional as F

from perf_model.tiny_m_dispatch import (
    tiny_m_linear, maybe_tiny_m_linear, dispatch_enabled, COUNTERS, ENABLE_ENV, THRESHOLD_ENV, OPS_ENV,
    ASSEMBLY_VARIANTS,
)

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
DTYPE = torch.float16 if CUDA else torch.float32  # fp16 matmul needs CUDA; CPU test path uses fp32

SHAPES = {
    "lm_head": (896, 151936), "qkv_proj": (896, 1152), "o_proj": (896, 896),
    "gate_up_proj": (896, 9728), "down_proj": (4864, 896),
}
M_VALUES = (1, 2, 3, 4, 6, 8, 9, 16)


def _make(K, N, M, bias, rank3=False):
    torch.manual_seed(0)
    W = (torch.randn(N, K, device=DEVICE, dtype=DTYPE) * 0.02)
    b = (torch.randn(N, device=DEVICE, dtype=DTYPE) * 0.02) if bias else None
    if rank3:
        x = torch.randn(1, M, K, device=DEVICE, dtype=DTYPE) * 0.02
    else:
        x = torch.randn(M, K, device=DEVICE, dtype=DTYPE) * 0.02
    return x, W, b


def _make_noncontiguous(K, N, M):
    """A real non-contiguous input: slice every other row out of a 2M-row
    tensor, exactly the kind of view vLLM can hand a linear layer."""
    torch.manual_seed(0)
    W = torch.randn(N, K, device=DEVICE, dtype=DTYPE) * 0.02
    x_full = torch.randn(2 * M, K, device=DEVICE, dtype=DTYPE) * 0.02
    x = x_full[::2]
    assert not x.is_contiguous()
    return x, W


@pytest.mark.parametrize("op_name,K,N", [(k, *v) for k, v in SHAPES.items()])
@pytest.mark.parametrize("M", M_VALUES)
def test_output_matches_flinear_within_fp16_tolerance(op_name, K, N, M):
    x, W, b = _make(K, N, M, bias=False)
    ref = F.linear(x, W, None)
    out = tiny_m_linear(x, W, None, threshold=8)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    assert out.device == ref.device
    max_abs = (out.float() - ref.float()).abs().max().item()
    max_rel = ((out.float() - ref.float()).abs() / ref.float().abs().clamp_min(1e-3)).max().item()
    assert max_abs < 0.05
    assert max_rel < 0.05


def test_bias_supported_and_correct():
    x, W, b = _make(896, 1152, 4, bias=True)
    ref = F.linear(x, W, b)
    out = tiny_m_linear(x, W, b, threshold=8)
    assert torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.05)


def test_noncontiguous_input_handled_correctly():
    x, W = _make_noncontiguous(896, 1152, 4)
    ref = F.linear(x, W, None)
    out = tiny_m_linear(x, W, None, threshold=8)
    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.05)


def test_rank3_input_reshaped_correctly():
    x, W, b = _make(896, 896, 4, bias=False, rank3=True)
    ref = F.linear(x, W, None)
    out = tiny_m_linear(x, W, None, threshold=8)
    assert out.shape == ref.shape == (1, 4, 896)
    assert torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.05)


@pytest.mark.parametrize("M", (9, 16))
def test_falls_back_to_default_above_threshold(M):
    x, W, b = _make(896, 1152, M, bias=False)
    out_default = F.linear(x, W, None)
    out_helper = tiny_m_linear(x, W, None, threshold=8)
    # above threshold, helper must be numerically identical to a direct F.linear call
    # (same code path), not merely close.
    assert torch.equal(out_helper, out_default)


def test_batch_one_uses_default_path_exactly():
    x, W, b = _make(896, 1152, 1, bias=False)
    out_default = F.linear(x, W, None)
    out_helper = tiny_m_linear(x, W, None, threshold=8)
    assert torch.equal(out_helper, out_default)


def test_deterministic_repeated_execution():
    x, W, b = _make(896, 9728, 4, bias=False)
    out1 = tiny_m_linear(x, W, None, threshold=8)
    out2 = tiny_m_linear(x, W, None, threshold=8)
    assert torch.equal(out1, out2)


def test_both_assembly_variants_produce_equivalent_output():
    x2d, W, b = _make(896, 1152, 4, bias=False)
    out_cat = ASSEMBLY_VARIANTS["cat"](x2d, W, b)
    out_prealloc = ASSEMBLY_VARIANTS["preallocate"](x2d, W, b)
    assert torch.equal(out_cat, out_prealloc)


# --- dispatch_enabled / env-gating (default-disabled contract) ---
def test_dispatch_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    enabled, _ = dispatch_enabled("lm_head")
    assert enabled is False


def test_dispatch_enabled_requires_both_flag_and_allowlist(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV, "1")
    monkeypatch.setenv(OPS_ENV, "gate_up")
    enabled, _ = dispatch_enabled("lm_head")
    assert enabled is False  # lm_head not in allowlist
    enabled2, _ = dispatch_enabled("gate_up")
    assert enabled2 is True


def test_dispatch_threshold_configurable(monkeypatch):
    monkeypatch.setenv(THRESHOLD_ENV, "4")
    _, threshold = dispatch_enabled("lm_head")
    assert threshold == 4


def test_maybe_tiny_m_linear_matches_flinear_when_disabled(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    x, W, b = _make(896, 1152, 4, bias=False)
    out = maybe_tiny_m_linear(x, W, None, op_family="lm_head")
    ref = F.linear(x, W, None)
    assert torch.equal(out, ref)


def test_maybe_tiny_m_linear_dispatches_when_enabled(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV, "1")
    monkeypatch.setenv(OPS_ENV, "lm_head")
    x, W, b = _make(896, 1152, 4, bias=False)
    out = maybe_tiny_m_linear(x, W, None, op_family="lm_head")
    ref = F.linear(x, W, None)
    assert torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.05)


# --- instrumentation counters ---
def test_instrumentation_counters_record_m_and_shape():
    COUNTERS.reset()
    x, W, b = _make(896, 1152, 4, bias=False)
    tiny_m_linear(x, W, None, threshold=8, instrument=True)
    snap = COUNTERS.snapshot()
    assert snap["optimized_calls"] == 1
    assert snap["observed_m"].get(4) == 1
    assert snap["observed_nk"].get("1152x896") == 1


def test_instrumentation_counters_record_fallback():
    COUNTERS.reset()
    x, W, b = _make(896, 1152, 9, bias=False)
    tiny_m_linear(x, W, None, threshold=8, instrument=True)
    snap = COUNTERS.snapshot()
    assert snap["fallback_calls"] == 1
    assert snap["optimized_calls"] == 0


def test_instrumentation_disabled_by_default_leaves_counters_untouched():
    COUNTERS.reset()
    x, W, b = _make(896, 1152, 4, bias=False)
    tiny_m_linear(x, W, None, threshold=8, instrument=False)
    snap = COUNTERS.snapshot()
    assert snap["optimized_calls"] == 0
    assert snap["fallback_calls"] == 0

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "cuda_transformer_kernels"
TOKENS_SWEEP = (1, 16, 128)
HIDDEN_SWEEP = (768, 1024, 4096, 8192)
RTOL = 1e-4
ATOL = 1e-4
EPS = 1e-6


def torch_rmsnorm(torch, x, weight, eps):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


@pytest.fixture(scope="session")
def torch_module():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch


@pytest.fixture(scope="session")
def rmsnorm_extension(torch_module):
    from torch.utils.cpp_extension import load

    return load(
        name="fused_rmsnorm_cuda_ext_test",
        sources=[
            str(KERNEL_DIR / "rmsnorm_extension.cpp"),
            str(KERNEL_DIR / "rmsnorm_kernel.cu"),
        ],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


@pytest.mark.parametrize("tokens", TOKENS_SWEEP)
@pytest.mark.parametrize("hidden", HIDDEN_SWEEP)
def test_fused_rmsnorm_cuda_matches_torch_fp32(torch_module, rmsnorm_extension, tokens, hidden):
    torch = torch_module
    torch.manual_seed(tokens * 100000 + hidden)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32)

    expected = torch_rmsnorm(torch, x, weight, EPS)
    actual = rmsnorm_extension.fused_rmsnorm_forward(x, weight, EPS)
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    assert torch.allclose(expected, actual, rtol=RTOL, atol=ATOL)


def test_fused_rmsnorm_cuda_rejects_non_contiguous_input(torch_module, rmsnorm_extension):
    torch = torch_module
    base = torch.randn(16, 4096, device="cuda", dtype=torch.float32)
    x = base[:, ::2]
    weight = torch.randn(x.shape[-1], device="cuda", dtype=torch.float32)

    assert not x.is_contiguous()
    with pytest.raises(RuntimeError, match="input must be contiguous"):
        rmsnorm_extension.fused_rmsnorm_forward(x, weight, EPS)

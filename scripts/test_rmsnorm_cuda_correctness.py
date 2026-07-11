"""RMSNorm CUDA correctness validation — block-size policy lab, Phase 1.

Validates the custom FP32 CUDA RMSNorm kernel against a PyTorch reference
across the full tokens x hidden x block_size matrix, reporting max absolute
error, max relative error, and pass/fail per case. Also validates that
unsupported block sizes and non-contiguous inputs are rejected with clear
errors.

SKIPs (exit 0) when PyTorch or CUDA is unavailable — correctness claims are
made only from an actual CUDA run.

Truth boundary: kernel-level correctness evidence only — not an end-to-end
Qwen or vLLM execution path.
"""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "cuda_transformer_kernels"

SUPPORTED_BLOCK_SIZES = (64, 128, 256, 512)


def torch_rmsnorm(torch, x, weight, eps):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,16,128")
    parser.add_argument("--hidden", default="768,1024,4096,8192")
    parser.add_argument("--block-sizes", default="64,128,256,512")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        print(f"SKIP: PyTorch import failed: {exc}")
        return 0

    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available")
        return 0

    extension = load(
        name="fused_rmsnorm_cuda_ext",
        sources=[
            str(KERNEL_DIR / "rmsnorm_extension.cpp"),
            str(KERNEL_DIR / "rmsnorm_kernel.cu"),
        ],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

    token_sizes = [int(item) for item in args.tokens.split(",") if item]
    hidden_sizes = [int(item) for item in args.hidden.split(",") if item]
    block_sizes = [int(item) for item in args.block_sizes.split(",") if item]

    reported = list(extension.supported_block_sizes())
    if reported != list(SUPPORTED_BLOCK_SIZES):
        raise SystemExit(
            f"extension reports supported block sizes {reported}, "
            f"expected {list(SUPPORTED_BLOCK_SIZES)}"
        )

    failures = []
    for tokens in token_sizes:
        for hidden in hidden_sizes:
            torch.manual_seed(tokens * 100000 + hidden)
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32)
            weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
            expected = torch_rmsnorm(torch, x, weight, args.eps)
            for block_size in block_sizes:
                actual = extension.fused_rmsnorm_forward(
                    x, weight, args.eps, block_size
                )
                torch.cuda.synchronize()
                abs_error = (expected - actual).abs()
                max_abs_error = abs_error.max().item()
                max_rel_error = (
                    (abs_error / expected.abs().clamp_min(1e-6)).max().item()
                )
                correct = bool(
                    torch.allclose(expected, actual, rtol=args.rtol, atol=args.atol)
                )
                finite = bool(torch.isfinite(actual).all())
                passed = correct and finite
                print(
                    f"tokens={tokens} hidden={hidden} block_size={block_size} "
                    f"max_abs_error={max_abs_error:.6g} "
                    f"max_rel_error={max_rel_error:.6g} "
                    f"finite={finite} pass={passed}"
                )
                if not passed:
                    failures.append(
                        (tokens, hidden, block_size, max_abs_error, max_rel_error)
                    )

    # Default-argument call path (block_size omitted -> 256, the original
    # launch configuration) must keep working for existing callers.
    x = torch.randn(4, 1024, device="cuda", dtype=torch.float32)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32)
    default_out = extension.fused_rmsnorm_forward(x, weight, args.eps)
    explicit_out = extension.fused_rmsnorm_forward(x, weight, args.eps, 256)
    torch.cuda.synchronize()
    if not torch.equal(default_out, explicit_out):
        failures.append(("default_block_size", "default != explicit 256"))
    else:
        print("default block_size call matches explicit block_size=256")

    # Unsupported block size must be rejected with a clear error.
    try:
        extension.fused_rmsnorm_forward(x, weight, args.eps, 96)
        failures.append(("unsupported_block_size", 96, "expected RuntimeError"))
    except RuntimeError as exc:
        if "unsupported block_size" not in str(exc):
            failures.append(("unsupported_block_size", 96, str(exc)))
        else:
            print("unsupported block_size=96 rejected as expected")

    # Non-contiguous input must be rejected with a clear error.
    base = torch.randn(16, 4096, device="cuda", dtype=torch.float32)
    non_contiguous = base[:, ::2]
    weight = torch.randn(non_contiguous.shape[-1], device="cuda", dtype=torch.float32)
    try:
        extension.fused_rmsnorm_forward(non_contiguous, weight, args.eps)
        failures.append(("non_contiguous_input", non_contiguous.shape, "expected RuntimeError"))
    except RuntimeError as exc:
        if "input must be contiguous" not in str(exc):
            failures.append(("non_contiguous_input", non_contiguous.shape, str(exc)))
        else:
            print("non_contiguous_input rejected as expected")

    if failures:
        raise SystemExit(f"RMSNorm correctness failed: {failures}")

    print("RMSNorm CUDA correctness passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

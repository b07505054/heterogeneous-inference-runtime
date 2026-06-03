import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "cuda_transformer_kernels"


def torch_rmsnorm(torch, x, weight, eps):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,16,128")
    parser.add_argument("--hidden", default="768,1024,4096,8192")
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

    failures = []
    for tokens in token_sizes:
        for hidden in hidden_sizes:
            torch.manual_seed(tokens * 100000 + hidden)
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32)
            weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
            expected = torch_rmsnorm(torch, x, weight, args.eps)
            actual = extension.fused_rmsnorm_forward(x, weight, args.eps)
            torch.cuda.synchronize()
            max_abs_error = (expected - actual).abs().max().item()
            correct = bool(torch.allclose(expected, actual, rtol=args.rtol, atol=args.atol))
            finite = bool(torch.isfinite(actual).all())
            print(
                f"tokens={tokens} hidden={hidden} "
                f"max_abs_error={max_abs_error:.6g} finite={finite} correct={correct}"
            )
            if not finite or not correct:
                failures.append((tokens, hidden, max_abs_error))

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

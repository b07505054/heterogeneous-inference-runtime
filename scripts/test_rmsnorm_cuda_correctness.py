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

    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

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
            print(
                f"tokens={tokens} hidden={hidden} "
                f"max_abs_error={max_abs_error:.6g} correct={correct}"
            )
            if not correct:
                failures.append((tokens, hidden, max_abs_error))

    if failures:
        raise SystemExit(f"RMSNorm correctness failed: {failures}")

    print("RMSNorm CUDA correctness passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

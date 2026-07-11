// PyTorch extension binding for the RMSNorm FP32 CUDA kernel.
//
// Block-size policy lab, Phase 1: callers may select a supported launch
// block size (64 / 128 / 256 / 512 threads per block). The default is 256
// — identical to the original fixed launch configuration, so existing
// callers are unaffected. Unsupported values are rejected with a clear
// error before any launch.
//
// Truth boundary: kernel-level correctness and microbenchmark evidence
// only — not an end-to-end Qwen or vLLM execution path.

#include <torch/extension.h>

#include <vector>

torch::Tensor fused_rmsnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps,
    int64_t block_size);

namespace {

const std::vector<int64_t> kSupportedBlockSizes = {64, 128, 256, 512};

bool is_supported_block_size(int64_t block_size) {
  for (int64_t supported : kSupportedBlockSizes) {
    if (block_size == supported) {
      return true;
    }
  }
  return false;
}

} // namespace

torch::Tensor fused_rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double eps,
    int64_t block_size) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must have shape [tokens, hidden]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden]");
  TORCH_CHECK(input.size(1) == weight.size(0), "weight hidden size mismatch");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(
      is_supported_block_size(block_size),
      "unsupported block_size ", block_size,
      "; supported block sizes: 64, 128, 256, 512");
  return fused_rmsnorm_forward_cuda(input, weight, eps, block_size);
}

std::vector<int64_t> supported_block_sizes() {
  return kSupportedBlockSizes;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fused_rmsnorm_forward",
      &fused_rmsnorm_forward,
      "Fused RMSNorm forward (CUDA)",
      pybind11::arg("input"),
      pybind11::arg("weight"),
      pybind11::arg("eps"),
      pybind11::arg("block_size") = 256);
  m.def(
      "supported_block_sizes",
      &supported_block_sizes,
      "Supported launch block sizes for fused_rmsnorm_forward");
}

#include <torch/extension.h>

torch::Tensor fused_rmsnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps);

torch::Tensor fused_rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double eps) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must have shape [tokens, hidden]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden]");
  TORCH_CHECK(input.size(1) == weight.size(0), "weight hidden size mismatch");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  return fused_rmsnorm_forward_cuda(input, weight, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fused_rmsnorm_forward",
      &fused_rmsnorm_forward,
      "Fused RMSNorm forward (CUDA)");
}

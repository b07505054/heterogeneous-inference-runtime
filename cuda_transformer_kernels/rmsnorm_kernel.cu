#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 256;

__inline__ __device__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__inline__ __device__ float block_reduce_sum(float value) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  value = warp_reduce_sum(value);

  if (lane == 0) {
    shared[warp_id] = value;
  }

  __syncthreads();

  value = threadIdx.x < blockDim.x / warpSize ? shared[lane] : 0.0f;

  if (warp_id == 0) {
    value = warp_reduce_sum(value);
  }

  if (threadIdx.x == 0) {
    shared[0] = value;
  }

  __syncthreads();

  return shared[0];
}

__global__ void rmsnorm_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int tokens,
    int hidden,
    float eps) {
  int row = blockIdx.x;
  if (row >= tokens) {
    return;
  }

  const float* row_input = input + row * hidden;
  float* row_output = output + row * hidden;

  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float x = row_input[col];
    sum_sq += x * x;
  }

  float total = block_reduce_sum(sum_sq);
  float inv_rms = rsqrtf(total / static_cast<float>(hidden) + eps);

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    row_output[col] = row_input[col] * inv_rms * weight[col];
  }
}

} // namespace

torch::Tensor fused_rmsnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps) {
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "only float32 is supported in this first kernel");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "only float32 weights are supported");

  auto output = torch::empty_like(input);
  int tokens = static_cast<int>(input.size(0));
  int hidden = static_cast<int>(input.size(1));

  rmsnorm_f32_kernel<<<tokens, kThreadsPerBlock>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      output.data_ptr<float>(),
      tokens,
      hidden,
      static_cast<float>(eps));

  return output;
}

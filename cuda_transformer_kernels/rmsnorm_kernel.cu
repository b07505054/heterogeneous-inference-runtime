// RMSNorm FP32 CUDA kernel — block-size policy lab, Phase 1.
//
// The BASELINE algorithm is deliberately preserved exactly:
//   - one CUDA block per token row,
//   - FP32 input, weight, accumulation, and output,
//   - scalar loads/stores (no float2/float4 vectorization),
//   - warp shuffle reduction + shared-memory block reduction,
//   - two-pass input access (read for sum-of-squares, read again for the
//     normalized write).
//
// Phase 1 changes ONLY the launch block size, as a controlled variable:
// compile-time specializations for 64 / 128 / 256 / 512 threads per block
// via a template + explicit dispatcher. 256 remains the default, matching
// the original fixed kThreadsPerBlock = 256 launch configuration.
//
// Truth boundary: kernel-level correctness and microbenchmark evidence
// only — not an end-to-end Qwen or vLLM execution path.

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kDefaultThreadsPerBlock = 256;

__inline__ __device__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <int kBlockSize>
__inline__ __device__ float block_reduce_sum(float value) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  value = warp_reduce_sum(value);

  if (lane == 0) {
    shared[warp_id] = value;
  }

  __syncthreads();

  value = threadIdx.x < kBlockSize / warpSize ? shared[lane] : 0.0f;

  if (warp_id == 0) {
    value = warp_reduce_sum(value);
  }

  if (threadIdx.x == 0) {
    shared[0] = value;
  }

  __syncthreads();

  return shared[0];
}

template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize) rmsnorm_f32_kernel(
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
  for (int col = threadIdx.x; col < hidden; col += kBlockSize) {
    float x = row_input[col];
    sum_sq += x * x;
  }

  float total = block_reduce_sum<kBlockSize>(sum_sq);
  float inv_rms = rsqrtf(total / static_cast<float>(hidden) + eps);

  for (int col = threadIdx.x; col < hidden; col += kBlockSize) {
    row_output[col] = row_input[col] * inv_rms * weight[col];
  }
}

template <int kBlockSize>
void launch_rmsnorm_f32(
    const float* input,
    const float* weight,
    float* output,
    int tokens,
    int hidden,
    float eps) {
  rmsnorm_f32_kernel<kBlockSize><<<tokens, kBlockSize>>>(
      input, weight, output, tokens, hidden, eps);
}

} // namespace

torch::Tensor fused_rmsnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps,
    int64_t block_size) {
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "only float32 is supported in this first kernel");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "only float32 weights are supported");

  auto output = torch::empty_like(input);
  int tokens = static_cast<int>(input.size(0));
  int hidden = static_cast<int>(input.size(1));

  const float* input_ptr = input.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  float* output_ptr = output.data_ptr<float>();
  float eps_f = static_cast<float>(eps);

  // Explicit compile-time block-size specializations. Unsupported values
  // are rejected with a clear error (also validated in the extension
  // binding before reaching this point).
  switch (block_size) {
    case 64:
      launch_rmsnorm_f32<64>(input_ptr, weight_ptr, output_ptr, tokens, hidden, eps_f);
      break;
    case 128:
      launch_rmsnorm_f32<128>(input_ptr, weight_ptr, output_ptr, tokens, hidden, eps_f);
      break;
    case 256:
      launch_rmsnorm_f32<256>(input_ptr, weight_ptr, output_ptr, tokens, hidden, eps_f);
      break;
    case 512:
      launch_rmsnorm_f32<512>(input_ptr, weight_ptr, output_ptr, tokens, hidden, eps_f);
      break;
    default:
      TORCH_CHECK(
          false,
          "unsupported block_size ", block_size,
          "; supported block sizes: 64, 128, 256, 512");
  }

  return output;
}

// Backward-compatible entry point: the original fixed launch configuration
// (256 threads per block) remains the default policy.
torch::Tensor fused_rmsnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps) {
  return fused_rmsnorm_forward_cuda(input, weight, eps, kDefaultThreadsPerBlock);
}

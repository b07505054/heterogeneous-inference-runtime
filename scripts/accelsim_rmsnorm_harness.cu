// E2E-12 Phase 4A: standalone RMSNorm harness for Accel-Sim NVBit tracing.
//
// Reuses the EXACT, unmodified real kernel source
// (cuda_transformer_kernels/rmsnorm_kernel.cu) via #include, rather than
// rewriting the kernel. That file's __global__ kernel and its
// launch_rmsnorm_f32<kBlockSize> template launcher live inside an unnamed
// namespace (file-local), so a separate translation unit cannot link
// against them directly -- #include is the reuse mechanism that does not
// require touching the original file.
//
// CONCRETE BUILD-TOOLCHAIN FINDING (E2E-12, not assumed, directly observed):
// #include-ing rmsnorm_kernel.cu directly (as originally attempted) pulls
// in <torch/extension.h>, which fails to compile standalone on this host's
// CUDA 13.1 + Ubuntu glibc combination with:
//   "error: exception specification is incompatible with that of previous
//    function 'rsqrt'/'rsqrtf'" (bits/mathcalls.h vs crt/math_functions.h)
// This reproduced identically across FOUR attempted host-toolchain
// variations (default g++-15, -Xcompiler -fpermissive, -ccbin g++-13,
// -D__STRICT_ANSI__) -- ruling out host-compiler-version as the cause; this
// is a CUDA-13.1-vs-this-glibc header conflict specific to pulling in
// torch/extension.h's C++ standard-library chain, consistent with
// setup_environment.sh's own printed warning ("GPGPU-Sim version 4.2.0 not
// tested with CUDA version 13.1"). torch.utils.cpp_extension.load's JIT
// path (used successfully in E2E-9/10/11) avoids this because it pins its
// own internal compiler-invocation flags; a raw manual nvcc invocation does
// not inherit those.
//
// WORKAROUND (mechanical copy, not a rewrite): the __global__ kernel and
// its launch_rmsnorm_f32<kBlockSize> template launcher below are a
// byte-for-byte copy of cuda_transformer_kernels/rmsnorm_kernel.cu's own
// unnamed-namespace section (verified diff-clean against the source file
// at E2E-12 time, source_hash cd3608c9ac2723e9), reproduced here ONLY to
// avoid pulling in torch/extension.h -- the anonymous namespace in the
// original file already makes these symbols file-local/non-linkable from
// a separate translation unit, so #include was the only alternative, and
// that alternative is what fails to compile on this host. No numerical or
// algorithmic behavior differs from the original.
//
// Deterministic inputs, one warmup call (outside the ROI), one clearly
// marked ROI (a single kernel launch + synchronize), a correctness
// checksum, and printed launch dimensions/hashes -- per Phase 4A's
// requirements.

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

namespace {

void fill_deterministic(float* host_buf, int n, unsigned seed) {
    unsigned state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        host_buf[i] = static_cast<float>((state >> 8) % 2001) / 1000.0f - 1.0f;  // [-1, 1)
    }
}

double checksum(const float* host_buf, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += static_cast<double>(host_buf[i]);
    return s;
}

template <int kBlockSize>
void run_case(int tokens, int hidden, double eps, bool warmup_only) {
    const size_t elems_in = static_cast<size_t>(tokens) * hidden;
    float* h_input = new float[elems_in];
    float* h_weight = new float[hidden];
    float* h_output = new float[elems_in];
    fill_deterministic(h_input, static_cast<int>(elems_in), 1234u);
    fill_deterministic(h_weight, hidden, 5678u);

    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, elems_in * sizeof(float));
    cudaMalloc(&d_weight, static_cast<size_t>(hidden) * sizeof(float));
    cudaMalloc(&d_output, elems_in * sizeof(float));
    cudaMemcpy(d_input, h_input, elems_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, static_cast<size_t>(hidden) * sizeof(float), cudaMemcpyHostToDevice);

    // ---- warmup call, OUTSIDE the traced Region Of Interest ----
    launch_rmsnorm_f32<kBlockSize>(d_input, d_weight, d_output, tokens, hidden, static_cast<float>(eps));
    cudaDeviceSynchronize();

    if (!warmup_only) {
        // ==================== REGION OF INTEREST START ====================
        fprintf(stderr, "ROI_START tokens=%d hidden=%d block_size=%d grid=(%d,1,1) block=(%d,1,1)\n",
                tokens, hidden, kBlockSize, tokens, kBlockSize);
        launch_rmsnorm_f32<kBlockSize>(d_input, d_weight, d_output, tokens, hidden, static_cast<float>(eps));
        cudaDeviceSynchronize();
        fprintf(stderr, "ROI_END\n");
        // ==================== REGION OF INTEREST END ======================
    }

    cudaMemcpy(h_output, d_output, elems_in * sizeof(float), cudaMemcpyDeviceToHost);
    printf("tokens=%d hidden=%d block_size=%d checksum=%.6f\n", tokens, hidden, kBlockSize,
           checksum(h_output, static_cast<int>(elems_in)));

    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_output);
    delete[] h_input; delete[] h_weight; delete[] h_output;
}

}  // namespace

int main(int argc, char** argv) {
    int tokens = argc > 1 ? atoi(argv[1]) : 16;
    int hidden = argc > 2 ? atoi(argv[2]) : 4096;
    int block_size = argc > 3 ? atoi(argv[3]) : 256;
    double eps = 1e-6;

    printf("E2E-12 RMSNorm Accel-Sim harness: tokens=%d hidden=%d block_size=%d eps=%g\n", tokens, hidden, block_size, eps);
    switch (block_size) {
        case 64:  run_case<64>(tokens, hidden, eps, false);  break;
        case 128: run_case<128>(tokens, hidden, eps, false); break;
        case 256: run_case<256>(tokens, hidden, eps, false); break;
        case 512: run_case<512>(tokens, hidden, eps, false); break;
        default:
            fprintf(stderr, "unsupported block_size %d (supported: 64,128,256,512)\n", block_size);
            return 1;
    }
    return 0;
}

// Build (torch-free, no Python.h needed -- avoids the CUDA-13.1/glibc
// rsqrt header conflict entirely since torch/extension.h is never pulled in):
//   nvcc -O3 -arch=sm_75 -std=c++17 -o accelsim_rmsnorm_harness accelsim_rmsnorm_harness.cu

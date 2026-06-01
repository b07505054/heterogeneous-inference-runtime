#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at line " << __LINE__ << "\n";     \
            std::exit(1);                                     \
        }                                                     \
    } while (0)

__global__ void matmul_naive_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void cpu_matmul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool validate(const float* cpu, const float* gpu, int N) {
    for (int i = 0; i < N * N; i++) {
        if (std::fabs(cpu[i] - gpu[i]) > 1e-2f) {
            std::cerr << "Mismatch at " << i
                      << ": cpu=" << cpu[i]
                      << ", gpu=" << gpu[i] << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 512;
    const size_t bytes = N * N * sizeof(float);

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C_cpu = new float[N * N];
    float* h_C_gpu = new float[N * N];

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matmul(h_A, h_B, h_C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Warmup
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    bool ok = validate(h_C_cpu, h_C_gpu, N);

    std::cout << "CUDA Naive MatMul\n";
    std::cout << "Matrix size: " << N << " x " << N << "\n";
    std::cout << "Block: " << block.x << " x " << block.y << "\n";
    std::cout << "Grid: " << grid.x << " x " << grid.y << "\n";
    std::cout << "CPU latency: " << cpu_ms << " ms\n";
    std::cout << "CUDA naive kernel latency: " << kernel_ms << " ms\n";
    std::cout << "Validation: " << (ok ? "PASSED" : "FAILED") << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    return 0;
}
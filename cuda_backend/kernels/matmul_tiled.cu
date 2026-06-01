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

constexpr int TILE = 16;

__global__ void matmul_tiled_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        int tiled_col = t * TILE + threadIdx.x;
        int tiled_row = t * TILE + threadIdx.y;

        if (row < N && tiled_col < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + tiled_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tiled_row < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiled_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
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

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Warmup
    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    bool ok = validate(h_C_cpu, h_C_gpu, N);

    std::cout << "CUDA Tiled Shared Memory MatMul\n";
    std::cout << "Matrix size: " << N << " x " << N << "\n";
    std::cout << "Tile size: " << TILE << " x " << TILE << "\n";
    std::cout << "Grid: " << grid.x << " x " << grid.y << "\n";
    std::cout << "CPU latency: " << cpu_ms << " ms\n";
    std::cout << "CUDA tiled kernel latency: " << kernel_ms << " ms\n";
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
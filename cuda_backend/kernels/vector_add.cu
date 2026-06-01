#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <vector>

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

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

double cpu_vector_add(const float* a, const float* b, float* c, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool validate(const float* c, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(c[i] - 3.0f) > 1e-5f) {
            return false;
        }
    }
    return true;
}

float run_cuda_kernel(
    const float* d_a,
    const float* d_b,
    float* d_c,
    int n,
    int block_size,
    int runs
) {
    int grid_size = (n + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 10; i++) {
        vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < runs; i++) {
        vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / runs;
}

int main() {
    const int n = 1 << 20;
    const int runs = 100;
    const size_t bytes = n * sizeof(float);

    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_cpu = new float[n];
    float* h_gpu = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    double cpu_ms = cpu_vector_add(h_a, h_b, h_cpu, n);

    float* d_a;
    float* d_b;
    float* d_c;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    std::vector<int> block_sizes = {64, 128, 256, 512, 1024};

    std::cout << "CUDA Vector Add Block Size Sweep\n";
    std::cout << "N elements: " << n << "\n";
    std::cout << "Runs per config: " << runs << "\n";
    std::cout << "CPU latency: " << cpu_ms << " ms\n\n";

    std::cout << "block_size,grid_size,kernel_avg_ms\n";

    for (int block_size : block_sizes) {
        int grid_size = (n + block_size - 1) / block_size;

        float kernel_ms = run_cuda_kernel(
            d_a,
            d_b,
            d_c,
            n,
            block_size,
            runs
        );

        std::cout << block_size << ","
                  << grid_size << ","
                  << kernel_ms << "\n";
    }

    CUDA_CHECK(cudaMemcpy(h_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    bool cpu_ok = validate(h_cpu, n);
    bool gpu_ok = validate(h_gpu, n);

    std::cout << "\nCPU validation: " << (cpu_ok ? "PASSED" : "FAILED") << "\n";
    std::cout << "GPU validation: " << (gpu_ok ? "PASSED" : "FAILED") << "\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    delete[] h_a;
    delete[] h_b;
    delete[] h_cpu;
    delete[] h_gpu;

    return 0;
}
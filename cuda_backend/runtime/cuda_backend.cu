#include "cuda_backend.h"

#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

std::string CUDABackend::name() const {
    return "CUDABackend";
}

void CUDABackend::vector_add(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c
) {
    int n = static_cast<int>(a.size());
    size_t bytes = n * sizeof(float);

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    vector_add_kernel<<<grid_size, block_size>>>(
        d_a,
        d_b,
        d_c,
        n
    );

    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
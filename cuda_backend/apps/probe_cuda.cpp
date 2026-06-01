#include <iostream>
#include <vector>

#include "../runtime/cuda_backend.h"

int main() {
    std::cout << "PROBE CUDA START" << std::endl;

    CUDABackend backend;

    std::vector<float> a(16, 1.0f);
    std::vector<float> b(16, 2.0f);
    std::vector<float> c(16, 0.0f);

    backend.vector_add(a, b, c);

    std::cout << "Backend: " << backend.name() << std::endl;

    for (float v : c) {
        std::cout << v << " ";
    }

    std::cout << std::endl;
    return 0;
}
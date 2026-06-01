#include <iostream>
#include <string>
#include <vector>

#include "../runtime/cpu_backend.h"
#include "../runtime/cuda_backend.h"

void run_cpu() {
    CPUBackend backend;

    std::vector<float> a(16, 1.0f);
    std::vector<float> b(16, 2.0f);
    std::vector<float> c(16, 0.0f);

    backend.vector_add(a, b, c);

    std::cout << "Backend: " << backend.name() << std::endl;

    for (float v : c) {
        std::cout << v << " ";
    }

    std::cout << std::endl;
}

void run_cuda() {
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
}

int main(int argc, char** argv) {
    std::cout << "[INFO] Backend dispatcher started" << std::endl;

    if (argc > 1 && std::string(argv[1]) == "cuda") {
        std::cout << "[INFO] Dispatching to CUDA backend" << std::endl;
        run_cuda();
    } else {
        std::cout << "[INFO] Dispatching to CPU backend" << std::endl;
        run_cpu();
    }

    std::cout << "[INFO] Backend dispatcher finished" << std::endl;
    return 0;
}
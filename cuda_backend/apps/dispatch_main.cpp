#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../runtime/backend.h"
#include "../runtime/cpu_backend.h"
#include "../runtime/cuda_backend.h"

int main(int argc, char** argv) {
    std::cout << "[INFO] dispatch_main started" << std::endl;

    bool use_cuda = false;

    if (argc > 1 && std::string(argv[1]) == "cuda") {
        use_cuda = true;
    }

    std::unique_ptr<Backend> backend;

    if (use_cuda) {
        std::cout << "[INFO] Selecting CUDA backend" << std::endl;
        backend = std::make_unique<CUDABackend>();
    } else {
        std::cout << "[INFO] Selecting CPU backend" << std::endl;
        backend = std::make_unique<CPUBackend>();
    }

    const int n = 16;

    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n, 0.0f);

    backend->vector_add(a, b, c);

    std::cout << "Backend: " << backend->name() << std::endl;
    std::cout << "Output:" << std::endl;

    for (float v : c) {
        std::cout << v << " ";
    }

    std::cout << std::endl;
    std::cout << "[INFO] dispatch_main finished" << std::endl;

    return 0;
}
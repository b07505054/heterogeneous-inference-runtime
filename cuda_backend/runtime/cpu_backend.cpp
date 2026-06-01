#include "cpu_backend.h"

std::string CPUBackend::name() const {
    return "CPUBackend";
}

void CPUBackend::vector_add(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c
) {
    int n = static_cast<int>(a.size());

    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
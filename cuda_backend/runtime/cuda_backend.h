#pragma once

#include "backend.h"

class CUDABackend : public Backend {
public:
    std::string name() const override;

    void vector_add(
        const std::vector<float>& a,
        const std::vector<float>& b,
        std::vector<float>& c
    ) override;
};
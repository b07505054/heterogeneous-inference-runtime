#pragma once

#include <vector>
#include <string>

class Backend {
public:
    virtual ~Backend() = default;

    virtual std::string name() const = 0;

    virtual void vector_add(
        const std::vector<float>& a,
        const std::vector<float>& b,
        std::vector<float>& c
    ) = 0;
};
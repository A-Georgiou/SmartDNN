#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include "Tensor.hpp"

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void optimize(const std::vector<std::reference_wrapper<Tensor>>& weights, const std::vector<std::reference_wrapper<Tensor>>& gradients, float learningRateOverride = -1.0f) = 0;

    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;
};

#endif // OPTIMIZER_HPP

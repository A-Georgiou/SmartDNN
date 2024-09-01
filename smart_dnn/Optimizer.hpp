#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include "Tensor/Tensor.hpp"
#include "Tensor/TensorConfig.hpp"

namespace smart_dnn {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void optimize(const std::vector<std::reference_wrapper<ConfiguredTensor<>>>& weights, const std::vector<std::reference_wrapper<ConfiguredTensor<>>>& gradients, float learningRateOverride = -1.0f) = 0;
    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;
};

} // namespace smart_dnn

#endif // OPTIMIZER_HPP

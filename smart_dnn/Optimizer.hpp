#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include "smart_dnn/tensor/TensorBase.hpp"

namespace sdnn {

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void optimize(
            const std::vector<std::reference_wrapper<Tensor>>& weights,
            const std::vector<std::reference_wrapper<Tensor>>& gradients,
            float learningRateOverride = -1.0) = 0;

    virtual inline void save(std::ostream& os) const { (void)os; };
    virtual inline void load(std::istream& is) { (void)is; };
};

} // namespace sdnn

#endif // OPTIMIZER_HPP

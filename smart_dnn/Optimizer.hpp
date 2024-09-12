#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include "tensor/Tensor.hpp"

namespace sdnn {

template <typename T=float>
class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void optimize(
            const std::vector<std::reference_wrapper<Tensor<T>>>& weights,
            const std::vector<std::reference_wrapper<Tensor<T>>>& gradients,
            T learningRateOverride = T(-1.0)) = 0;

    virtual inline void save(std::ostream& os) const { (void)os; };
    virtual inline void load(std::istream& is) { (void)is; };
};

} // namespace sdnn

#endif // OPTIMIZER_HPP

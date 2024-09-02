#ifndef LOSS_HPP
#define LOSS_HPP

#include "Tensor/Tensor.hpp"

namespace smart_dnn {

template <typename T>
class Loss {
public:
    virtual ~Loss() = default;
    virtual Tensor<T> compute(const Tensor<T>& prediction, const Tensor<T>& target) = 0;
    virtual Tensor<T> gradient(const Tensor<T>& prediction, const Tensor<T>& target) = 0;
    
    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;
};

} // namespace smart_dnn

#endif // LOSS_HPP
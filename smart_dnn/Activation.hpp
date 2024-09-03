#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "Tensor/Tensor.hpp"

namespace smart_dnn {

template <typename T>
class Activation {
public:
    virtual ~Activation() = default;
    virtual Tensor<T> forward(const Tensor<T>& input) const = 0;
    virtual Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const= 0;
};

} // namespace smart_dnn

#endif // ACTIVATION_HPP
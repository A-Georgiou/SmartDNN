#ifndef TANH_HPP
#define TANH_HPP

#include <cmath>
#include "smart_dnn/Activation.hpp"
#include "smart_dnn/tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

/*

    Tanh Activation Function
    ------------------------

    f(x) = (exp(2x) - 1) / (exp(2x) + 1)
    f'(x) = 1 - tanh(x)^2

*/
template <typename T>
class Tanh : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return AdvancedTensorOperations<T>::apply(input, [](T x) { return (std::exp(x * T(2)) - T(1)) / (std::exp(x * T(2)) + T(1)); });
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        return AdvancedTensorOperations<T>::apply(input, [](T x) {  return T(1) - std::tanh(x) * std::tanh(x);  }) * gradOutput;
    }
};

} // namespace smart_dnn

#endif // TANH_HPP
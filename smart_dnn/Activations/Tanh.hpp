#ifndef TANH_HPP
#define TANH_HPP

#include <cmath>
#include "../Activation.hpp"
#include "../Tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

template <typename T>
class Tanh : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return AdvancedTensorOperations::apply(input, [](T x) { return (std::exp(x * T(2)) - T(1)) / (std::exp(x * T(2)) + T(1)); });
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        return AdvancedTensorOperations::apply(input, [](T x) {  return T(1) - std::tanh(x) * std::tanh(x);  }) * gradOutput;
    }
};

} // namespace smart_dnn

#endif // TANH_HPP
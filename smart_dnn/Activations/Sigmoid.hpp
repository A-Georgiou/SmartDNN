#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "smart_dnn/Activation.hpp"
#include "smart_dnn/tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

/*

    Sigmoid Activation Function
    ------------------------

    f(x) = 1 / (1 + exp(-x))
    f'(x) = f(x) * (1 - f(x))

*/
template <typename T=float>
class Sigmoid : public Activation<T> {
    using TensorType = Tensor<T>;
public:
    TensorType forward(const TensorType& input) const override {
        return AdvancedTensorOperations<T>::apply(input, [](float x) { return T(1) / (T(1) + std::exp(-x)); });
    }

    TensorType backward(const TensorType& input, const TensorType& gradOutput) const override {
        TensorType sig = forward(input);
        return sig * (T(1) - sig) * gradOutput;
    }
};

} // namespace smart_dnn

#endif // SIGMOID_HPP
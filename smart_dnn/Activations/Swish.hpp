#ifndef SWISH_HPP
#define SWISH_HPP

#include <cmath>
#include "smart_dnn/Activation.hpp"
#include "smart_dnn/Tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

/*

    Swish Activation Function
    ------------------------

    f(x) = x * sigmoid(x) = x / (1 + exp(-x))
    f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

*/
template <typename T = float>
class Swish : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return AdvancedTensorOperations<T>::apply(input, [](T x) { 
            T sigmoid_x = T(1) / (T(1) + std::exp(-x));
            return x * sigmoid_x; 
        });
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        return AdvancedTensorOperations<T>::applyPair(input, gradOutput, [](T x, T grad) {
            T sigmoid_x = T(1) / (T(1) + std::exp(-x));
            T derivative = sigmoid_x * (T(1) + x * (T(1) - sigmoid_x));
            return grad * derivative;
        });
    }
};

} // namespace smart_dnn

#endif // SWISH_HPP
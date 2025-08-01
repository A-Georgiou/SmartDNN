#ifndef MISH_HPP
#define MISH_HPP

#include <cmath>
#include "smart_dnn/Activation.hpp"
#include "smart_dnn/Tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

/*

    Mish Activation Function
    ------------------------

    f(x) = x * tanh(softplus(x))
         = x * tanh(ln(1 + exp(x)))

    f'(x) = tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)
           where sech²(y) = 1 - tanh²(y)

*/
template <typename T = float>
class Mish : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return AdvancedTensorOperations<T>::apply(input, [](T x) { 
            T softplus_x = std::log(T(1) + std::exp(x));
            return x * std::tanh(softplus_x); 
        });
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        return AdvancedTensorOperations<T>::applyPair(input, gradOutput, [](T x, T grad) {
            T softplus_x = std::log(T(1) + std::exp(x));
            T tanh_softplus = std::tanh(softplus_x);
            T sigmoid_x = T(1) / (T(1) + std::exp(-x));
            T sech_squared = T(1) - tanh_softplus * tanh_softplus;
            T derivative = tanh_softplus + x * sech_squared * sigmoid_x;
            return grad * derivative;
        });
    }
};

} // namespace smart_dnn

#endif // MISH_HPP
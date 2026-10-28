#ifndef MISH_HPP
#define MISH_HPP

#include <algorithm>
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
            return x * std::tanh(softplus(x));
        });
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        return AdvancedTensorOperations<T>::applyPair(input, gradOutput, [](T x, T grad) {
            T softplus_x = softplus(x);
            T tanh_softplus = std::tanh(softplus_x);
            T exp_neg_abs_x = std::exp(-std::abs(x));
            T sigmoid_x = x >= T(0)
                ? T(1) / (T(1) + exp_neg_abs_x)
                : exp_neg_abs_x / (T(1) + exp_neg_abs_x);
            T sech_squared = T(1) - tanh_softplus * tanh_softplus;
            T derivative = tanh_softplus + x * sech_squared * sigmoid_x;
            return grad * derivative;
        });
    }

private:
    static T softplus(T x) {
        return std::max(x, T(0)) + std::log1p(std::exp(-std::abs(x)));
    }
};

} // namespace smart_dnn

#endif // MISH_HPP
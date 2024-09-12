#ifndef LEAKY_RELU_HPP
#define LEAKY_RELU_HPP

#include "smart_dnn/Activation.hpp"
#include "smart_dnn/tensor/Tensor.hpp"
#include "smart_dnn/tensor/AdvancedTensorOperations.hpp"
#include <algorithm>

namespace sdnn {

/*

    Leaky ReLU Activation Function
    ------------------------

    f(x) = x if x > 0, alpha * x otherwise
    f'(x) = 1 if x > 0, alpha otherwise

*/
template <typename T = float>
class LeakyReLU : public Activation<T> {
public:
    explicit LeakyReLU(T alpha = T(0.01)) : alpha(alpha) {}

    Tensor<T> forward(const Tensor<T>& input) const override {
        Tensor<T> output(input);
        return AdvancedTensorOperations<T>::apply(input, [this](T x) { return (x > 0) ? x : (alpha * x); });
    }

   Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        return AdvancedTensorOperations<T>::applyPair(input, gradOutput, 
            [this](T x, T grad) { return (x >= T(0)) ? grad : (alpha * grad); });
    }

private:
    T alpha;
};

} // namespace sdnn

#endif // LEAKY_RELU_HPP

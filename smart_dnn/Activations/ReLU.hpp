#ifndef RELU_HPP
#define RELU_HPP

#include "smart_dnn/Activation.hpp"
#include "smart_dnn/tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

/*

    ReLU Activation Function
    ------------------------

    f(x) = max(0, x)
    f'(x) = 1 if x > 0, 0 otherwise

*/
template <typename T=float>
class ReLU : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return AdvancedTensorOperations<T>::apply(input, [](T x) { return std::max(T(0), x); });
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        return AdvancedTensorOperations<T>::apply(input, [](T x) { return x > T(0) ? T(1) : T(0); }) * gradOutput;
    }
};

}

#endif // RELU_HPP
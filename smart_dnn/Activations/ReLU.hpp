#ifndef RELU_HPP
#define RELU_HPP

#include "../Activation.hpp"
#include "../Tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

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
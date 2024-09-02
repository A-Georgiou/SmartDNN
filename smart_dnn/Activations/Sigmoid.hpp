#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "../Activation.hpp"
#include "../Tensor/AdvancedTensorOperations.hpp"

template <typename T>
class Sigmoid : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return AdvancedTensorOperations::apply(input, [](float x) { return T(1) / (T(1) + std::exp(-x)); });
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        Tensor<T> sig = forward(input);
        return sig * (T(1) - sig) * gradOutput;
    }
};

#endif // SIGMOID_HPP
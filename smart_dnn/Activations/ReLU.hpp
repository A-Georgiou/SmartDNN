#ifndef RELU_HPP
#define RELU_HPP

#include "../Activation.hpp"
#include "../TensorOperations.hpp"

class ReLU : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        return input.apply([](float x) { return std::max(0.0f, x); });
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        return input.apply([](float x) { return x > 0.0f ? 1.0f : 0.0f; }) * gradOutput;
    }
};

#endif // RELU_HPP
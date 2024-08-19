#ifndef TANH_HPP
#define TANH_HPP

#include "../smart_dnn/Activation.hpp"

class Tanh : public Activation {
    public:
    Tensor forward(const Tensor& input) const override {
        return input.apply([](float x) { return (std::exp(x*2)-1)/(std::exp(x*2)+1); });
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        return input.apply([](float x) { return 1.0f - std::tanh(x) * std::tanh(x); }) * gradOutput;
    }
};

#endif // TANH_HPP
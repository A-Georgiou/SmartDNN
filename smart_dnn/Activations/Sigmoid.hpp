#include "../smart_dnn/Activation.hpp"

class Sigmoid : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        return input.apply([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        Tensor sig = forward(input);
        return sig * (1.0f - sig) * gradOutput;
    }
};
#ifndef LEAKY_RELU_HPP
#define LEAKY_RELU_HPP

#include "../smart_dnn/Activation.hpp"
#include "../Tensor.hpp"

class LeakyReLU : public Activation {
public:
    explicit LeakyReLU(float alpha = 0.01f) : alpha(alpha) {}

    Tensor forward(const Tensor& input) const override {
        Tensor output = input;
        for (size_t i = 0; i < output.getData().size(); ++i) {
            output.getData()[i] = (input.getData()[i] > 0) ? input.getData()[i] : alpha * input.getData()[i];
        }
        return output;
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        Tensor gradInput = input;
        for (size_t i = 0; i < gradInput.getData().size(); ++i) {
            gradInput.getData()[i] = (input.getData()[i] > 0) ? gradOutput.getData()[i] : alpha * gradOutput.getData()[i];
        }
        return gradInput;
    }

private:
    float alpha;
};

#endif // LEAKY_RELU_HPP
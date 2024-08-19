#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../smart_dnn/Activation.hpp"
#include "../Tensor.hpp"

class Softmax : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        Tensor output = input;

        float maxVal = *std::max_element(output.getData().begin(), output.getData().end());

        float sum = 0.0f;
        for (float& val : output.getData()) {
            val = std::exp(val - maxVal);
            sum += val;
        }

        for (float& val : output.getData()) {
            val /= sum;
        }

        return output;
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        Tensor output = forward(input);
        Tensor gradInput(input.shape());

        for (size_t i = 0; i < gradInput.getData().size(); ++i) {
            float gradient = 0.0f;
            for (size_t j = 0; j < gradInput.getData().size(); ++j) {
                if (i == j) {
                    gradient += output.getData()[i] * (1.0f - output.getData()[i]) * gradOutput.getData()[j];
                } else {
                    gradient -= output.getData()[i] * output.getData()[j] * gradOutput.getData()[j];
                }
            }
            gradInput.getData()[i] = gradient;
        }

        return gradInput;
    }
};

#endif // SOFTMAX_HPP

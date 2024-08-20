#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../smart_dnn/Activation.hpp"
#include "../Tensor.hpp"

class Softmax : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        Tensor output = input;

        float maxVal = *std::max_element(output.getData().begin(), output.getData().end());
        output -= maxVal;
        output = output.apply([](float x) { return std::exp(x); });
        float sum = std::accumulate(output.getData().begin(), output.getData().end(), 0.0f);
        output /= sum;

        return output;
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        Tensor output = forward(input);
        Tensor gradInput(input.shape());

        for (size_t i = 0; i < gradInput.getData().size(); ++i) {
            float softmaxI = output.getData()[i];
            float gradient = 0.0f;
            for (size_t j = 0; j < gradInput.getData().size(); ++j) {
                float softmaxJ = output.getData()[j];
                if (i == j) {
                    gradient += softmaxI * (1.0f - softmaxJ) * gradOutput.getData()[j];
                } else {
                    gradient -= softmaxI * softmaxJ * gradOutput.getData()[j];
                }
            }
            gradInput.getData()[i] = gradient;
        }

        return gradInput;
    }
};

#endif // SOFTMAX_HPP

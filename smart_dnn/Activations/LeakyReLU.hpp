#ifndef LEAKY_RELU_HPP
#define LEAKY_RELU_HPP

#include "../Activation.hpp"
#include "../Tensor.hpp"
#include <algorithm>

class LeakyReLU : public Activation {
public:
    explicit LeakyReLU(float alpha = 0.01f) : alpha(alpha) {}

    Tensor forward(const Tensor& input) const override {
        Tensor output(input.shape());
        const float* inputData = input.getData();
        float* outputData = output.getData();
        int size = input.shape().size();

        for (int i = 0; i < size; ++i) {
            outputData[i] = (inputData[i] > 0) ? inputData[i] : alpha * inputData[i];
        }

        return output;
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        Tensor gradInput(input.shape());
        const float* inputData = input.getData();
        const float* gradOutputData = gradOutput.getData();
        float* gradInputData = gradInput.getData();
        int size = input.shape().size();

        for (int i = 0; i < size; ++i) {
            gradInputData[i] = (inputData[i] > 0) ? gradOutputData[i] : alpha * gradOutputData[i];
        }

        return gradInput;
    }

private:
    float alpha;
};

#endif // LEAKY_RELU_HPP
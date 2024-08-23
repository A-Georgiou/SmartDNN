#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../smart_dnn/Activation.hpp"
#include "../Tensor.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

class Softmax : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        Tensor output(input.shape());
        const float* inputData = input.getData();
        float* outputData = output.getData();
        int size = input.shape().size();

        float maxVal = *std::max_element(inputData, inputData + size);

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            outputData[i] = std::exp(inputData[i] - maxVal);
            sum += outputData[i];
        }

        for (int i = 0; i < size; ++i) {
            outputData[i] /= sum;
        }

        return output;
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        Tensor forwardOutput = forward(input);
        Tensor gradInput(input.shape());

        const float* outputData = forwardOutput.getData();
        const float* gradOutputData = gradOutput.getData();
        float* gradInputData = gradInput.getData();
        int size = input.shape().size();

        for (int i = 0; i < size; ++i) {
            float softmaxI = outputData[i];
            float gradient = 0.0f;
            for (int j = 0; j < size; ++j) {
                float softmaxJ = outputData[j];
                if (i == j) {
                    gradient += softmaxI * (1.0f - softmaxJ) * gradOutputData[j];
                } else {
                    gradient -= softmaxI * softmaxJ * gradOutputData[j];
                }
            }
            gradInputData[i] = gradient;
        }

        return gradInput;
    }
};

#endif // SOFTMAX_HPP
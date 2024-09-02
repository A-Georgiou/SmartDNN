#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../Activation.hpp"
#include "../Tensor/Tensor.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

template <typename T>
class Softmax : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        Tensor<T> output(input.shape());
        const T* inputData = input.getData();
        T* outputData = output.getData();
        int size = input.shape().size();

        T maxVal = *std::max_element(inputData, inputData + size);

        float sum = T(0);
        for (int i = 0; i < size; ++i) {
            outputData[i] = std::exp(inputData[i] - maxVal);
            sum += outputData[i];
        }

        for (int i = 0; i < size; ++i) {
            outputData[i] /= sum;
        }

        return output;
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        Tensor<T> forwardOutput = forward(input);
        Tensor<T> gradInput(input.shape());

        const T* outputData = forwardOutput.getData();
        const T* gradOutputData = gradOutput.getData();
        T* gradInputData = gradInput.getData();
        int size = input.shape().size();

        for (int i = 0; i < size; ++i) {
            T softmaxI = outputData[i];
            T gradient = T(0);
            for (int j = 0; j < size; ++j) {
                T softmaxJ = outputData[j];
                if (i == j) {
                    gradient += softmaxI * (T(1) - softmaxJ) * gradOutputData[j];
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
#ifndef LEAKY_RELU_HPP
#define LEAKY_RELU_HPP

#include "../Activation.hpp"
#include "../Tensor/Tensor.hpp"
#include <algorithm>

template <typename T = float>
class LeakyReLU : public Activation<T> {
public:
    explicit LeakyReLU(T alpha = T(0.01)) : alpha(alpha) {}

    Tensor<T> forward(const Tensor<T>& input) const override {
        Tensor<T> output(input.shape());
        const T* inputData = input.getData();
        T* outputData = output.getData();
        int size = input.shape().size();

        for (int i = 0; i < size; ++i) {
            outputData[i] = (inputData[i] > 0) ? inputData[i] : alpha * inputData[i];
        }

        return output;
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        Tensor<T> gradInput(input.shape());
        const T* inputData = input.getData();
        const T* gradOutputData = gradOutput.getData();
        T* gradInputData = gradInput.getData();
        int size = input.shape().size();

        for (int i = 0; i < size; ++i) {
            gradInputData[i] = (inputData[i] > 0) ? gradOutputData[i] : alpha * gradOutputData[i];
        }

        return gradInput;
    }

private:
    T alpha;
};

#endif // LEAKY_RELU_HPP

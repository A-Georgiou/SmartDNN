#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../Activation.hpp"
#include "../Tensor/Tensor.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace smart_dnn {

template <typename T=float>
class Softmax : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        Tensor<T> output(input.getShape());
        const T* inputData = input.getData().data();
        T* outputData = output.getData().data();
        int size = input.getShape().size();

        T maxVal = *std::max_element(inputData, inputData + size);
        T sum = T(0);

        for (int i = 0; i < size; ++i) {
            outputData[i] = std::exp(inputData[i] - maxVal);
            sum += outputData[i];
        }

        return output.apply([sum](T x) { return x / sum; });
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        Tensor<T> forwardOutput = forward(input);
        Tensor<T> gradInput(input.getShape());

        const T* outputData = forwardOutput.getData().data();
        const T* gradOutputData = gradOutput.getData().data();
        T* gradInputData = gradInput.getData().data();
        int size = input.getShape().size();

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

} // namespace smart_dnn

#endif // SOFTMAX_HPP
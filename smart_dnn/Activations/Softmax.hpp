#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../Activation.hpp"
#include "../Tensor/Tensor.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

namespace smart_dnn {

template <typename T=float>
class Softmax : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        if (input.getShape().rank() < 2) {
            throw std::invalid_argument("Input must have at least 2 dimensions (batch_size, features)");
        }

        Tensor<T> output(input.getShape());
        const T* inputData = input.getData().data();
        T* outputData = output.getData().data();

        int batchSize = input.getShape()[0];
        int featuresSize = input.getShape().size() / batchSize;

        for (int b = 0; b < batchSize; ++b) {
            const T* batchInputData = inputData + b * featuresSize;
            T* batchOutputData = outputData + b * featuresSize;

            T maxVal = *std::max_element(batchInputData, batchInputData + featuresSize);
            T sum = T(0);

            for (int i = 0; i < featuresSize; ++i) {
                batchOutputData[i] = std::exp(batchInputData[i] - maxVal);
                sum += batchOutputData[i];
            }

            for (int i = 0; i < featuresSize; ++i) {
                batchOutputData[i] /= sum;
            }
        }


        return output;
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        if (input.getShape().rank() < 2 || gradOutput.getShape().rank() < 2) {
            throw std::invalid_argument("Input and gradOutput must have at least 2 dimensions (batch_size, features)");
        }
        
        Tensor<T> forwardOutput = forward(input);
        Tensor<T> gradInput(input.getShape());

        const T* outputData = forwardOutput.getData().data();
        const T* gradOutputData = gradOutput.getData().data();
        T* gradInputData = gradInput.getData().data();

        int batchSize = input.getShape()[0];
        int featuresSize = input.getShape().size() / batchSize;

        for (int b = 0; b < batchSize; ++b) {
            const T* batchOutputData = outputData + b * featuresSize;
            const T* batchGradOutputData = gradOutputData + b * featuresSize;
            T* batchGradInputData = gradInputData + b * featuresSize;

            for (int i = 0; i < featuresSize; ++i) {
                T gradient = T(0);
                for (int j = 0; j < featuresSize; ++j) {
                    if (i == j) {
                        gradient += batchOutputData[i] * (T(1) - batchOutputData[j]) * batchGradOutputData[j];
                    } else {
                        gradient -= batchOutputData[i] * batchOutputData[j] * batchGradOutputData[j];
                    }
                }
                batchGradInputData[i] = gradient;
            }
        }

        return gradInput;
    }
};

} // namespace smart_dnn

#endif // SOFTMAX_HPP
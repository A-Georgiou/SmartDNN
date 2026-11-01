#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "smart_dnn/Activation.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

namespace smart_dnn {


/*

    Softmax Activation Function
    ---------------------------
    
    f(x) = exp(x) / sum(exp(x))
    f'(x) = f(x) * (1 - f(x))

*/
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

        const size_t features = input.getShape()[input.getShape().rank() - 1];
        const size_t rows = input.getShape().size() / features;

        // One softmax per row of the last axis.
        for (size_t r = 0; r < rows; ++r) {
            const T* rowInputData = inputData + r * features;
            T* rowOutputData = outputData + r * features;

            T maxVal = *std::max_element(rowInputData, rowInputData + features);
            T sum = T(0);

            for (size_t i = 0; i < features; ++i) {
                rowOutputData[i] = std::exp(rowInputData[i] - maxVal);
                sum += rowOutputData[i];
            }

            for (size_t i = 0; i < features; ++i) {
                rowOutputData[i] /= sum;
            }
        }


        return output;
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        if (input.getShape().rank() < 2 || gradOutput.getShape() != input.getShape()) {
            throw std::invalid_argument("gradOutput must have the input's shape, with at least 2 dimensions (batch_size, features)");
        }
        
        Tensor<T> forwardOutput = forward(input);
        Tensor<T> gradInput(input.getShape());

        const T* outputData = forwardOutput.getData().data();
        const T* gradOutputData = gradOutput.getData().data();
        T* gradInputData = gradInput.getData().data();

        const size_t features = input.getShape()[input.getShape().rank() - 1];
        const size_t rows = input.getShape().size() / features;

        for (size_t r = 0; r < rows; ++r) {
            const T* rowOutputData = outputData + r * features;
            const T* rowGradOutputData = gradOutputData + r * features;
            T* rowGradInputData = gradInputData + r * features;

            // dx = y * (dy - sum(dy * y)) is the row's Jacobian product without forming the Jacobian.
            T dot = T(0);
            for (size_t i = 0; i < features; ++i) {
                dot += rowGradOutputData[i] * rowOutputData[i];
            }

            for (size_t i = 0; i < features; ++i) {
                rowGradInputData[i] = rowOutputData[i] * (rowGradOutputData[i] - dot);
            }
        }

        return gradInput;
    }
};

} // namespace smart_dnn

#endif // SOFTMAX_HPP

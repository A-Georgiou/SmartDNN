#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "smart_dnn/Activation.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

namespace sdnn {

/*

    Softmax Activation Function
    ---------------------------
    
    f(x) = exp(x) / sum(exp(x))
    f'(x) = f(x) * (1 - f(x))

*/
class Softmax : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        /*
        if (input.shape().rank() < 2) {
            throw std::invalid_argument("Input must have at least 2 dimensions (batch_size, features)");
        }

        Tensor output = input.clone(); // Create a copy of the input tensor to store the output

        int batchSize = input.shape()[0];
        int featuresSize = input.shape().size() / batchSize;

        for (int b = 0; b < batchSize; ++b) {
            std::vector<size_t> batchIndices = {static_cast<size_t>(b)};
            Tensor batchInput = input[batchIndices];  // Slice the batch from input
            Tensor batchOutput = output[batchIndices]; // Slice the batch from output

            // Find the maximum value in the batch for numerical stability
            double maxVal = sum(batchInput, {0}, false).at<double>(0); // Assuming sum can act as a reduction op to find max
            Tensor maxTensor = fill(batchInput.shape(), maxVal, input.type());
            Tensor shiftedInput = batchInput - maxTensor;

            Tensor expInput = exp(shiftedInput);  // exp(input - maxVal)
            double sumExp = sum(expInput, {0}, false).at<double>(0); // Sum of exponentials
            Tensor sumExpTensor = fill(expInput.shape(), sumExp, input.type());

            // Normalize the result: output = expInput / sum(expInput)
            batchOutput = div(expInput, sumExpTensor);
        }

        return output;
        */
       return input;
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        /*
        if (input.shape().rank() < 2 || gradOutput.shape().rank() < 2) {
            throw std::invalid_argument("Input and gradOutput must have at least 2 dimensions (batch_size, features)");
        }

        Tensor forwardOutput = forward(input);
        Tensor gradInput(input.shape(), input.type());

        int batchSize = input.shape()[0];
        int featuresSize = input.shape().size() / batchSize;

        for (int b = 0; b < batchSize; ++b) {
            std::vector<size_t> batchIndices = {b};
            Tensor batchOutput = forwardOutput[batchIndices];
            Tensor batchGradOutput = gradOutput[batchIndices];
            Tensor batchGradInput = gradInput[batchIndices];

            auto backend = input.backend();

            // Apply the softmax gradient formula:
            // gradInput = output * (gradOutput - sum(gradOutput * output))
            Tensor outputMulGrad = backend.mul(batchOutput, batchGradOutput);
            double sumGradOutput = backend.sum(outputMulGrad, {0}, false).at<double>(0);
            Tensor sumGradTensor = backend.fill(batchOutput.shape(), sumGradOutput, input.type());

            batchGradInput = backend.mul(batchOutput, backend.sub(batchGradOutput, sumGradTensor));
        }

        return gradInput;
        */
       return gradOutput;
    }
};

} // namespace sdnn

#endif // SOFTMAX_HPP
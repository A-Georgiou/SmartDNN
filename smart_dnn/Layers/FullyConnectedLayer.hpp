#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

#include "../Tensor.hpp"
#include "../Optimizer.hpp"
#include "../TensorOperations.hpp"
#include "../Layer.hpp"
#include "../TensorWrapper.hpp"
#include <vector>

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer(int inputSize, int outputSize) {
        weights = Tensor({inputSize, outputSize});
        biases = Tensor({1, outputSize});
        (*weights).randomize(-1.0, 1.0);
        (*biases).fill(0.0);
    }

    Tensor forward(Tensor& input) override {
        this->input = input;
        Tensor output = TensorOperations::matmul(input, *weights);  
        output += *biases;
        return output;
    }

    Tensor backward(Tensor& gradOutput) override {
        Tensor transposedInput = *input;
        Tensor& weightsTensor = *weights;

        if (transposedInput.shape().rank() > 1) {
            transposedInput = TensorOperations::transpose(transposedInput, 0, 1);
        }

        weightGradients = TensorOperations::matmul(transposedInput, gradOutput);
        biasGradients = gradOutput.sum(0);

        Tensor weightsTransposed(weightsTensor.shape());
        if (weightsTransposed.shape().rank() > 1) {
            weightsTransposed = TensorOperations::transpose(weightsTensor, 0, 1);
        } else {
            weightsTransposed = weightsTensor; 
        }

        return TensorOperations::matmul(gradOutput, weightsTransposed);
    }

    void updateWeights(Optimizer& optimizer) override {
        // Pass references to avoid copies
        optimizer.optimize({std::ref(*weights), std::ref(*biases)}, {std::ref(*weightGradients), std::ref(*biasGradients)});
    }

private:
    TensorWrapper weights;
    TensorWrapper biases;
    TensorWrapper input;
    TensorWrapper weightGradients;
    TensorWrapper biasGradients;
};

#endif // FULLY_CONNECTED_LAYER_HPP

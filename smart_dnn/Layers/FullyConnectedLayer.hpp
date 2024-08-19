#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

#include "../Tensor.hpp"
#include "../Optimizer.hpp"
#include "../TensorOperations.hpp"
#include "../Layer.hpp"
#include <vector>

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer(int inputSize, int outputSize) {
        weights = Tensor({inputSize, outputSize});
        biases = Tensor({1, outputSize});
        weights.randomize(-1.0, 1.0);
        biases.fill(0.0);
    }

    Tensor forward(Tensor& input) override {
        this->input = input;
        Tensor output = TensorOperations::matmul(input, weights);  
        output.add(biases);
        return output;
    }

    Tensor backward(Tensor& gradOutput) override {
        Tensor transposedInput = input;

        if (input.shape().rank() > 1) {
            transposedInput = TensorOperations::transpose(input, 0, 1);
        }

        weightGradients = TensorOperations::matmul(transposedInput, gradOutput);
        biasGradients = gradOutput.sum(0);

        Tensor weightsTransposed;
        if (weights.shape().rank() > 1) {
            weightsTransposed = TensorOperations::transpose(weights, 0, 1);
        } else {
            weightsTransposed = weights; 
        }

        return TensorOperations::matmul(gradOutput, weightsTransposed);
    }

    void updateWeights(Optimizer& optimizer) override {
        // Pass references to avoid copies
        optimizer.optimize({std::ref(weights), std::ref(biases)}, {std::ref(weightGradients), std::ref(biasGradients)});
    }

private:
    Tensor weights;
    Tensor biases;
    Tensor input;
    Tensor weightGradients;
    Tensor biasGradients;
};

#endif // FULLY_CONNECTED_LAYER_HPP

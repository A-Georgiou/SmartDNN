#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/Optimizer.hpp"
#include "smart_dnn/Layer.hpp"
#include <vector>
#include <optional>

namespace sdnn {

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer(int inputSize, int outputSize) {
        weights.emplace(rand(Shape({inputSize, outputSize})));
        biases.emplace(zeros(Shape({1, outputSize})));
    }

    /*
    
    Fully Connected Layer forward pass
    -------------------------------

    Input: 1D or 2D tensor shape (input_size) or (batch_size, input_size)
    Output: 1D or 2D tensor shape (output_size) or (batch_size, output_size)
    
    */
    
    Tensor forward(const Tensor& input) override {
        this->input = input;
        Tensor reshapedInput = input;
        if (input.shape().rank() == 1) {
            reshapedInput = reshape(input, {1, input.shape()[0]});
        }

        Tensor output = matmul(reshapedInput, *weights);
        output = output + *biases;

        if (input.shape().rank() == 1) {
            output = reshape(output, {output.shape()[1]});
        }

        return output;
    }

    /*

    Fully Connected Layer backward pass
    -------------------------------

    Input: Gradient tensor shape (batch_size, output_size)
    Output: Gradient tensor shape (batch_size, input_size)

    */
    Tensor backward(const Tensor& gradOutput) override {
        if (!input) {
            throw std::runtime_error("Input tensor is not initialized!");
        }

        Tensor inputTensor = *input;

        // Ensure input is 2D
        if (inputTensor.shape().rank() == 1)
            inputTensor = reshape(inputTensor, {1, inputTensor.shape()[0]});

        // Ensure gradOutput is 2D (for non-batched input case)
        Tensor reshapedGradOutput = gradOutput;
        if (gradOutput.shape().rank() == 1) {
            reshapedGradOutput = reshape(gradOutput, {1, gradOutput.shape()[0]});
        }

        // Calculate weight gradients
        Tensor inputTransposed = transpose(inputTensor, {1, 0}); // Shape: (input_size, batch_size)

        weightGradients.emplace(matmul(inputTransposed, reshapedGradOutput)); // Shape: (input_size, output_size)

        // Calculate bias gradients
        biasGradients.emplace(reshape(sum(reshapedGradOutput, {0}), {1, biases->shape()[1]}));

        // Calculate input gradients
        Tensor weightsTransposed = transpose(*weights, {1, 0}); // Shape: (output_size, input_size)

        Tensor gradInput = matmul(reshapedGradOutput, weightsTransposed); // Shape: (batch_size, input_size)

        return gradInput;
    }

    void updateWeights(Optimizer& optimizer) override {
        if (!weights || !biases || !weightGradients || !biasGradients) {
            throw std::runtime_error("Weights or gradients are not initialized!");
        }

        Tensor oldWeights = (*weights).clone();
        Tensor oldBiases = (*biases).clone();
        Tensor oldWeightGradients = (*weightGradients).clone();
        Tensor oldBiasGradients = (*biasGradients).clone();

        optimizer.optimize({std::ref(*weights), std::ref(*biases)},
                           {std::ref(*weightGradients), std::ref(*biasGradients)});

        std::cout << "Updated weights: " << ((*weights) - oldWeights).toString() << std::endl;
        std::cout << "Updated biases: " << ((*biases) - oldBiases).toString() << std::endl;
        std::cout << "weight gradients: " << oldWeightGradients.toString() << std::endl;
        std::cout << "bias gradients: " << oldBiasGradients.toString() << std::endl;
    }

    /*
    
        Test helper functions
    
    */

    Tensor getWeights() const {
        return *weights;
    }

    Tensor getBiases() const {
        return *biases;
    }

    Tensor getWeightGradients() const {
        return *weightGradients;
    }

    Tensor getBiasGradients() const {
        return *biasGradients;
    }

    void setWeights(const Tensor& newWeights) {
        weights = newWeights;
    }

    void setBiases(const Tensor& newBiases) {
        if (newBiases.shape().rank() != 2 || newBiases.shape()[0] != 1) {
            throw std::invalid_argument("Biases must have shape (1, outputSize)");
        }
        biases = newBiases;
    }

    void setBiasGradient(const Tensor& newBiasGradient) {
        biasGradients = newBiasGradient;
    }

    void setWeightGradient(const Tensor& newWeightGradient) {
        weightGradients = newWeightGradient;
    }

private:
    std::optional<Tensor> weights;
    std::optional<Tensor> biases;
    std::optional<Tensor> input;
    std::optional<Tensor> weightGradients;
    std::optional<Tensor> biasGradients;
};

} // namespace sdnn

#endif // FULLY_CONNECTED_LAYER_HPP

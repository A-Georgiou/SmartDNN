#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

#include "smart_dnn/tensor/Tensor.hpp"
#include "smart_dnn/tensor/AdvancedTensorOperations.hpp"
#include "smart_dnn/Optimizer.hpp"
#include "smart_dnn/Layer.hpp"
#include <vector>
#include <optional>

namespace smart_dnn {

template <typename T = float>
class FullyConnectedLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    FullyConnectedLayer(int inputSize, int outputSize) {
        weights.emplace(TensorType::rand(Shape({inputSize, outputSize})));
        biases.emplace(TensorType::zeros(Shape({1, outputSize})));
    }

    /*
    
    Fully Connected Layer forward pass
    -------------------------------

    Input: 1D or 2D tensor shape (input_size) or (batch_size, input_size)
    Output: 1D or 2D tensor shape (output_size) or (batch_size, output_size)
    
    */
    
    TensorType forward(const TensorType& input) override {
        this->input = input;
        TensorType reshapedInput = input;
        if (input.getShape().rank() == 1) {
            reshapedInput = AdvancedTensorOperations<T>::reshape(input, {1, input.getShape()[0]});
        }
        TensorType output = AdvancedTensorOperations<T>::matmul(reshapedInput, *weights);
        output = output + *biases;

        if (input.getShape().rank() == 1) {
            output = AdvancedTensorOperations<T>::reshape(output, {output.getShape()[1]});
        }
        return output;
    }

    /*

    Fully Connected Layer backward pass
    -------------------------------

    Input: Gradient tensor shape (batch_size, output_size)
    Output: Gradient tensor shape (batch_size, input_size)

    */
    TensorType backward(const TensorType& gradOutput) override {
        if (!input) {
            throw std::runtime_error("Input tensor is not initialized!");
        }

        TensorType inputTensor = *input;

        // Ensure input is 2D
        if (inputTensor.getShape().rank() == 1)
            inputTensor = AdvancedTensorOperations<T>::reshape(inputTensor, {1, inputTensor.getShape()[0]});

        // Ensure gradOutput is 2D (for non-batched input case)
        TensorType reshapedGradOutput = gradOutput;
        if (gradOutput.getShape().rank() == 1) {
            reshapedGradOutput = AdvancedTensorOperations<T>::reshape(gradOutput, {1, gradOutput.getShape()[0]});
        }

        // Calculate weight gradients
        TensorType inputTransposed = AdvancedTensorOperations<T>::transpose(inputTensor, 0, 1); // Shape: (input_size, batch_size)
        weightGradients.emplace(AdvancedTensorOperations<T>::matmul(inputTransposed, reshapedGradOutput)); // Shape: (input_size, output_size)

        // Calculate bias gradients
        biasGradients.emplace(AdvancedTensorOperations<T>::reshape(
            AdvancedTensorOperations<T>::sum(reshapedGradOutput, 0),
            {1, biases->getShape()[1]}
        ));

        // Calculate input gradients
        TensorType weightsTransposed = AdvancedTensorOperations<T>::transpose(*weights, 0, 1); // Shape: (output_size, input_size)
        TensorType gradInput = AdvancedTensorOperations<T>::matmul(reshapedGradOutput, weightsTransposed); // Shape: (batch_size, input_size)

        return gradInput;
    }

    void updateWeights(Optimizer<T>& optimizer) override {
        if (!weights || !biases || !weightGradients || !biasGradients) {
            throw std::runtime_error("Weights or gradients are not initialized!");
        }

        optimizer.optimize({std::ref(*weights), std::ref(*biases)},
                           {std::ref(*weightGradients), std::ref(*biasGradients)});
    }

    /*
    
        Test helper functions
    
    */

    TensorType getWeights() const {
        return *weights;
    }

    TensorType getBiases() const {
        return *biases;
    }

    TensorType getWeightGradients() const {
        return *weightGradients;
    }

    TensorType getBiasGradients() const {
        return *biasGradients;
    }

    void setWeights(const TensorType& newWeights) {
        weights = newWeights;
    }

    void setBiases(const TensorType& newBiases) {
        if (newBiases.getShape().rank() != 2 || newBiases.getShape()[0] != 1) {
            throw std::invalid_argument("Biases must have shape (1, outputSize)");
        }
        biases = newBiases;
    }

    void setBiasGradient(const TensorType& newBiasGradient) {
        biasGradients = newBiasGradient;
    }

    void setWeightGradient(const TensorType& newWeightGradient) {
        weightGradients = newWeightGradient;
    }

private:
    std::optional<TensorType> weights;
    std::optional<TensorType> biases;
    std::optional<TensorType> input;
    std::optional<TensorType> weightGradients;
    std::optional<TensorType> biasGradients;
};

} // namespace smart_dnn

#endif // FULLY_CONNECTED_LAYER_HPP

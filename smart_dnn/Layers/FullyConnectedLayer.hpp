#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

#include "../Tensor/Tensor.hpp"
#include "../Tensor/AdvancedTensorOperations.hpp"
#include "../Optimizer.hpp"
#include "../Layer.hpp"
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
    
    TensorType forward(const TensorType& input) override {
        this->input = input;
        TensorType output = AdvancedTensorOperations<T>::matmul(input, *weights);
        TensorType biasBroadcast = AdvancedTensorOperations<T>::reshape(*biases, {1, biases->getShape()[1]});
        output = output + biasBroadcast;
        return output;
    }

    TensorType backward(const TensorType& gradOutput) override {
        if (!input) {
            throw std::runtime_error("Input tensor is not initialized!");
        }

        TensorType inputTensor = *input;

        if (inputTensor.getShape().rank() == 1)
            inputTensor = AdvancedTensorOperations<T>::reshape(inputTensor, {1, inputTensor.getShape()[0]});

        TensorType inputTransposed = AdvancedTensorOperations<T>::transpose(inputTensor, 0, 1); // Shape: (input_size, batch_size)
        weightGradients.emplace(AdvancedTensorOperations<T>::matmul(inputTransposed, gradOutput)); // (input_size, output_size)

        biasGradients.emplace(AdvancedTensorOperations<T>::sum(gradOutput, 0));

        TensorType weightsTransposed = AdvancedTensorOperations<T>::transpose(*weights, 0, 1); // Shape: (output_size, input_size)
        TensorType gradInput = AdvancedTensorOperations<T>::matmul(gradOutput, weightsTransposed); // (batch_size, input_size)

        return gradInput;
    }

    void updateWeights(Optimizer<T>& optimizer) override {
        if (!weights || !biases || !weightGradients || !biasGradients) {
            throw std::runtime_error("Weights or gradients are not initialized!");
        }

        optimizer.optimize({std::ref(*weights), std::ref(*biases)},
                           {std::ref(*weightGradients), std::ref(*biasGradients)});
    }

    void setTrainingMode(bool mode) override {
        trainingMode = mode;
    }

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

    bool trainingMode = true;
};

} // namespace smart_dnn

#endif // FULLY_CONNECTED_LAYER_HPP

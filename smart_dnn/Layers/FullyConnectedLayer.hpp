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
        TensorType output = AdvancedTensorOperations<T>::matmul(this->input.value(), weights.value()); 
        output += biases.value();
        return output;
    }

    TensorType backward(const TensorType& gradOutput) override {
        if (!input) {
            throw std::runtime_error("Input tensor is not initialized!");
        }

        TensorType inputTransposed = input.value();
        if (inputTransposed.getShape().rank() > 1) {
            inputTransposed = AdvancedTensorOperations<T>::transpose(input.value(), 0, 1);
        }

        weightGradients.emplace(AdvancedTensorOperations<T>::matmul(inputTransposed, gradOutput));
        biasGradients.emplace(AdvancedTensorOperations<T>::sum(gradOutput));

        TensorType weightsTransposed = weights.value();
        if (weightsTransposed.getShape().rank() > 1) {
            weightsTransposed = AdvancedTensorOperations<T>::transpose(weights.value(), 0, 1);
        }

        return AdvancedTensorOperations<T>::matmul(gradOutput, weightsTransposed);
    }

    void updateWeights(Optimizer<T>& optimizer) override {
        if (!weights || !biases || !weightGradients || !biasGradients) {
            throw std::runtime_error("Weights or gradients are not initialized!");
        }

        optimizer.optimize({std::ref(weights.value()), std::ref(biases.value())},
                           {std::ref(weightGradients.value()), std::ref(biasGradients.value())});
    }

    void setTrainingMode(bool mode) override {
        trainingMode = mode;
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

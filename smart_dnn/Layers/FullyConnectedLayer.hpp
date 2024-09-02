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
        return AdvancedTensorOperations<T>::matmul(this->input.value(), weights.value()) + biases.value();
    }

    TensorType backward(const TensorType& gradOutput) override {
        if (!input) {
            throw std::runtime_error("Input tensor is not initialized!");
        }

        TensorType gradOutputReshaped = gradOutput;
        TensorType inputTensor = input.value();
        TensorType weightsTransposed = weights.value();

        if (inputTensor.getShape().rank() == 1)
            inputTensor = AdvancedTensorOperations<T>::reshape(inputTensor, {1, inputTensor.getShape()[0]});

        if (gradOutput.getShape().rank() == 1 && gradOutput.getShape()[0] == 1)
            gradOutputReshaped = AdvancedTensorOperations<T>::reshape(gradOutput, {1, 1});

        TensorType inputTransposed = AdvancedTensorOperations<T>::transpose(inputTensor, 0, 1);
        weightGradients.emplace(AdvancedTensorOperations<T>::matmul(inputTransposed, gradOutputReshaped));
        biasGradients.emplace(AdvancedTensorOperations<T>::sum(gradOutputReshaped));

        if (weightsTransposed.getShape().rank() > 1)
            weightsTransposed = AdvancedTensorOperations<T>::transpose(weights.value(), 0, 1);

        return AdvancedTensorOperations<T>::matmul(gradOutputReshaped, weightsTransposed);
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

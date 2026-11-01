#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Tensor/AdvancedTensorOperations.hpp"
#include "smart_dnn/Optimizer.hpp"
#include "smart_dnn/Layer.hpp"
#include "smart_dnn/RandomEngine.hpp"
#include <vector>
#include <optional>

namespace smart_dnn {

template <typename T = float>
class FullyConnectedLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    FullyConnectedLayer(int inputSize, int outputSize) {
        weights.emplace(Shape({inputSize, outputSize}));
        for (T& w : weights->getData()) w = static_cast<T>(RandomEngine::getHeInit(inputSize));
        biases.emplace(TensorType::zeros(Shape({1, outputSize})));
        weightGradients.emplace(noGradient<T>());
        biasGradients.emplace(noGradient<T>());
    }

    /*
    
    Fully Connected Layer forward pass
    -------------------------------

    Input: tensor of shape (..., input_size); each index of the leading dimensions is one row
    Output: tensor of shape (..., output_size)
    
    */
    
    TensorType forward(const TensorType& input) override {
        this->input = input;
        std::vector<int> outputDimensions = input.getShape().getDimensions();
        outputDimensions.back() = weights->getShape()[1];

        TensorType output = AdvancedTensorOperations<T>::matmul(asRows(input), *weights);
        output = output + *biases;
        output.reshape(outputDimensions);
        return output;
    }

    /*

    Fully Connected Layer backward pass
    -------------------------------

    Input: Gradient tensor shape (..., output_size)
    Output: Gradient tensor with the input's shape (..., input_size)

    */
    TensorType backward(const TensorType& gradOutput) override {
        if (!input) {
            throw std::runtime_error("Input tensor is not initialized!");
        }

        TensorType inputRows = asRows(*input); // Shape: (rows, input_size)
        TensorType gradOutputRows = asRows(gradOutput); // Shape: (rows, output_size)

        // Calculate weight gradients
        TensorType inputTransposed = AdvancedTensorOperations<T>::transpose(inputRows, 0, 1); // Shape: (input_size, rows)
        weightGradients.emplace(AdvancedTensorOperations<T>::matmul(inputTransposed, gradOutputRows)); // Shape: (input_size, output_size)

        // Calculate bias gradients
        biasGradients.emplace(AdvancedTensorOperations<T>::reshape(
            AdvancedTensorOperations<T>::sum(gradOutputRows, 0),
            {1, biases->getShape()[1]}
        ));

        // Calculate input gradients
        TensorType weightsTransposed = AdvancedTensorOperations<T>::transpose(*weights, 0, 1); // Shape: (output_size, input_size)
        TensorType gradInput = AdvancedTensorOperations<T>::matmul(gradOutputRows, weightsTransposed); // Shape: (rows, input_size)
        gradInput.reshape(input->getShape());

        return gradInput;
    }

    std::vector<Parameter<T>> parameters() override {
        return {{*weights, *weightGradients}, {*biases, *biasGradients}};
    }

    void save(std::ostream& os) const override {
        saveTensor(os, *weights);
        saveTensor(os, *biases);
    }

    void load(std::istream& is) override {
        loadTensor(is, *weights);
        loadTensor(is, *biases);
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
    // Views (..., features) as (rows, features), so a rank-1 tensor is a single row.
    static TensorType asRows(const TensorType& tensor) {
        const Shape& shape = tensor.getShape();
        const int features = shape[shape.rank() - 1];
        return AdvancedTensorOperations<T>::reshape(tensor, {static_cast<int>(shape.size()) / features, features});
    }

    std::optional<TensorType> weights;
    std::optional<TensorType> biases;
    std::optional<TensorType> input;
    std::optional<TensorType> weightGradients;
    std::optional<TensorType> biasGradients;
};

} // namespace smart_dnn

#endif // FULLY_CONNECTED_LAYER_HPP

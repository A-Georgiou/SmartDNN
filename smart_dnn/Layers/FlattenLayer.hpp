#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include "../Layer.hpp"
#include "../Tensor/Tensor.hpp"
#include "../Tensor/AdvancedTensorOperations.hpp"

template <typename T>
class FlattenLayer : public Layer {
    using TensorType = Tensor<T>;
public:
    TensorType forward(TensorType& input) override {
        if (input.shape().rank() < 2) {
            throw std::invalid_argument("Input tensor must have at least 2 dimensions");
        }

        if (input.shape().rank() == 2) {
            return input;
        }

        this->originalShape = input.shape();
        int batchSize = this->originalShape[0];
        int flattenedSize = this->originalShape.size() / batchSize;
        
        return AdvancedTensorOperations::reshape(input, Shape({batchSize, flattenedSize}));
    }

    TensorType backward(TensorType& gradOutput) override {
        return AdvancedTensorOperations::reshape(gradOutput, originalShape);
    }

    void updateWeights(Optimizer<T>& optimizer) override {
        // No weights to update in a flatten layer.
    }
private:
    Shape originalShape;
};

#endif // FLATTEN_LAYER_HPP
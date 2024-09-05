#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include "../Layer.hpp"
#include "../Tensor/Tensor.hpp"
#include "../Tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

template <typename T=float>
class FlattenLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    FlattenLayer() {};

    TensorType forward(const TensorType& input) override {
        if (input.getShape().rank() < 2) {
            throw std::invalid_argument("Input tensor must have at least 2 dimensions");
        }

        if (input.getShape().rank() == 2) {
            return input;
        }

        this->originalShape = input.getShape();
        int batchSize = (*originalShape)[0];
        int flattenedSize = (*originalShape).size() / batchSize;
        
        return AdvancedTensorOperations<T>::reshape(input, Shape({batchSize, flattenedSize}));
    }

    TensorType backward(const TensorType& gradOutput) override {
        return AdvancedTensorOperations<T>::reshape(gradOutput, (*originalShape));
    }

    void updateWeights(Optimizer<T>& optimizer) override {
        // No weights to update in a flatten layer.
    }

    void setTrainingMode(bool mode) override {
        // No training mode in a flatten layer.
    }

private:
    std::optional<Shape> originalShape;
};

} // namespace smart_dnn

#endif // FLATTEN_LAYER_HPP
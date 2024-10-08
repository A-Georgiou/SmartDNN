#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/tensor/Tensor.hpp"
#include "smart_dnn/tensor/AdvancedTensorOperations.hpp"

namespace smart_dnn {

template <typename T=float>
class FlattenLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    FlattenLayer() = default;

    /*
    
    Flatten Layer forward pass
    -------------------------------

    Input: 2D or higher dimensional tensor, shape (batch_size, ...)
    Output: 2D tensor, shape (batch_size, flattened_size)

    */
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

    /*
    
    Flatten Layer backward pass
    -------------------------------

    Input: Gradient tensor, shape (batch_size, flattened_size)
    Output: Gradient tensor, shape (batch_size, ...)
    
    */
    TensorType backward(const TensorType& gradOutput) override {
        return AdvancedTensorOperations<T>::reshape(gradOutput, (*originalShape));
    }

private:
    std::optional<Shape> originalShape;
};

} // namespace smart_dnn

#endif // FLATTEN_LAYER_HPP
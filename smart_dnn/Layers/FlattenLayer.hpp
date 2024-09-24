#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"

namespace sdnn {

class FlattenLayer : public Layer {
public:
    FlattenLayer() = default;

    /*
    
    Flatten Layer forward pass
    -------------------------------

    Input: 2D or higher dimensional tensor, shape (batch_size, ...)
    Output: 2D tensor, shape (batch_size, flattened_size)

    */
    Tensor forward(const Tensor& input) override {
        if (input.shape().rank() < 2) {
            throw std::invalid_argument("Input tensor must have at least 2 dimensions");
        }

        if (input.shape().rank() == 2) {
            return input;
        }

        this->originalShape = input.shape();
        int batchSize = (*originalShape)[0];
        int flattenedSize = (*originalShape).size() / batchSize;
        
        return reshape(input, Shape({batchSize, flattenedSize}));
    }

    /*
    
    Flatten Layer backward pass
    -------------------------------

    Input: Gradient tensor, shape (batch_size, flattened_size)
    Output: Gradient tensor, shape (batch_size, ...)
    
    */
    Tensor backward(const Tensor& gradOutput) override {
        return reshape(gradOutput, (*originalShape));
    }

private:
    std::optional<Shape> originalShape;
};

} // namespace sdnn

#endif // FLATTEN_LAYER_HPP
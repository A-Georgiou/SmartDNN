#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include "../Layer.hpp"
#include "../Tensor.hpp"
#include "../TensorOperations.hpp"

class FlattenLayer : public Layer {
    public:
        Tensor forward(Tensor& input) override {
            if (input.shape().rank() < 2) {
                throw std::invalid_argument("Input tensor must have at least 2 dimensions");
            }

            if (input.shape().rank() == 2) {
                return input;
            }

            this->originalShape = input.shape().getDimensions();

            int batchSize = input.shape()[0];
            int flattenedSize = input.shape().size() / batchSize;

            return TensorOperations::reshape(input, Shape({batchSize, flattenedSize}));
        }

        Tensor backward(Tensor& gradOutput) override {
            return TensorOperations::reshape(gradOutput, originalShape);
        }

        void updateWeights(Optimizer& optimizer) override {
            // No weights to update in a flatten layer.
        }
    private:
        std::vector<int> originalShape;
};

#endif // FLATTEN_LAYER_HPP
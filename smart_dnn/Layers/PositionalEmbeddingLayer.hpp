#ifndef POSITIONAL_EMBEDDING_LAYER_HPP
#define POSITIONAL_EMBEDDING_LAYER_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/RandomEngine.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include <algorithm>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace smart_dnn {

// Adds a learned vector for each position to inputs of shape (..., length, embeddingSize), for lengths up to maxLength.
template <typename T = float>
class PositionalEmbeddingLayer : public Layer<T> {
public:
    PositionalEmbeddingLayer(int maxLength, int embeddingSize)
        : weights(Shape({maxLength, embeddingSize})), weightGradients(Shape({maxLength, embeddingSize}), T(0)) {
        std::normal_distribution<float> distribution(0.0f, 0.02f);
        for (T& weight : weights.getData()) {
            weight = static_cast<T>(distribution(RandomEngine::getInstance()));
        }
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        const Shape& shape = input.getShape();
        const size_t rank = shape.rank();
        if (rank < 2 || shape[rank - 1] != weights.getShape()[1] || shape[rank - 2] > weights.getShape()[0]) {
            throw std::invalid_argument("Positional embedding needs (..., length, " + std::to_string(weights.getShape()[1])
                + ") with length up to " + std::to_string(weights.getShape()[0]) + ", got " + shape.toString());
        }

        inputShape = shape;
        sequenceSize = static_cast<size_t>(shape[rank - 2]) * shape[rank - 1];
        Tensor<T> output = input;
        T* values = output.getData().data();
        const T* table = weights.getData().data();
        for (size_t start = 0; start < shape.size(); start += sequenceSize) {
            for (size_t i = 0; i < sequenceSize; ++i) {
                values[start + i] += table[i];
            }
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        if (!inputShape || gradOutput.getShape() != *inputShape) {
            throw std::invalid_argument("Positional embedding backward needs gradOutput with the last forward pass's shape");
        }

        T* table = weightGradients.getData().data();
        std::fill(table, table + weightGradients.getShape().size(), T(0));
        const T* values = gradOutput.getData().data();
        for (size_t start = 0; start < gradOutput.getShape().size(); start += sequenceSize) {
            for (size_t i = 0; i < sequenceSize; ++i) {
                table[i] += values[start + i];
            }
        }
        return gradOutput;
    }

    std::vector<Parameter<T>> parameters() override {
        return {{weights, weightGradients}};
    }

    void save(std::ostream& os) const override {
        saveTensor(os, weights);
    }

    void load(std::istream& is) override {
        loadTensor(is, weights);
    }

private:
    Tensor<T> weights;
    Tensor<T> weightGradients;
    std::optional<Shape> inputShape;
    size_t sequenceSize = 0;
};

} // namespace smart_dnn

#endif // POSITIONAL_EMBEDDING_LAYER_HPP

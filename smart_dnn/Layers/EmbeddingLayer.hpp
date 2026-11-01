#ifndef EMBEDDING_LAYER_HPP
#define EMBEDDING_LAYER_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/RandomEngine.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace smart_dnn {

// Maps token ids, stored as T in any shape (...), to learned rows (..., embeddingSize); the ids get no gradient.
template <typename T = float>
class EmbeddingLayer : public Layer<T> {
public:
    EmbeddingLayer(int vocabularySize, int embeddingSize)
        : weights(Shape({vocabularySize, embeddingSize})), weightGradients(Shape({vocabularySize, embeddingSize}), T(0)) {
        std::normal_distribution<float> distribution(0.0f, 0.02f);
        for (T& weight : weights.getData()) {
            weight = static_cast<T>(distribution(RandomEngine::getInstance()));
        }
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        const T vocabularySize = T(weights.getShape()[0]);
        const size_t width = weights.getShape()[1];
        ids.resize(input.getShape().size());
        for (size_t i = 0; i < ids.size(); ++i) {
            const T id = input[i];
            if (!(id >= T(0) && id < vocabularySize) || id != std::floor(id)) {
                throw std::invalid_argument("Embedding ids must be whole numbers below the vocabulary size");
            }
            ids[i] = static_cast<size_t>(id);
        }

        inputDimensions = input.getShape().getDimensions();
        std::vector<int> outputDimensions = inputDimensions;
        outputDimensions.push_back(static_cast<int>(width));
        Tensor<T> output(Shape(outputDimensions), T(0));
        const T* table = weights.getData().data();
        T* rows = output.getData().data();
        for (size_t i = 0; i < ids.size(); ++i) {
            std::copy(table + ids[i] * width, table + (ids[i] + 1) * width, rows + i * width);
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        const size_t width = weights.getShape()[1];
        std::vector<int> outputDimensions = inputDimensions;
        outputDimensions.push_back(static_cast<int>(width));
        if (ids.empty() || gradOutput.getShape().getDimensions() != outputDimensions) {
            throw std::invalid_argument("Embedding backward needs gradOutput with the last forward pass's output shape");
        }

        T* table = weightGradients.getData().data();
        std::fill(table, table + weightGradients.getShape().size(), T(0));
        const T* rows = gradOutput.getData().data();
        for (size_t i = 0; i < ids.size(); ++i) {
            for (size_t j = 0; j < width; ++j) {
                table[ids[i] * width + j] += rows[i * width + j];
            }
        }
        return Tensor<T>(Shape(inputDimensions), T(0));
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
    std::vector<size_t> ids;
    std::vector<int> inputDimensions;
};

} // namespace smart_dnn

#endif // EMBEDDING_LAYER_HPP

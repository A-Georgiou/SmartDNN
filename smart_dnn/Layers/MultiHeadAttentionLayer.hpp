#ifndef MULTI_HEAD_ATTENTION_LAYER_HPP
#define MULTI_HEAD_ATTENTION_LAYER_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Tensor/AdvancedTensorOperations.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace smart_dnn {

// Self-attention over (batch, length, embeddingSize) with several heads; when causal, each position only attends to itself and earlier ones.
template <typename T = float>
class MultiHeadAttentionLayer : public Layer<T> {
    using Ops = AdvancedTensorOperations<T>;

public:
    MultiHeadAttentionLayer(int embeddingSize, int heads, bool causal = true)
        : heads(heads), causal(causal), inputProjection(embeddingSize, 3 * embeddingSize), outputProjection(embeddingSize, embeddingSize) {
        if (heads <= 0 || embeddingSize % heads != 0) {
            throw std::invalid_argument("embeddingSize must be a positive multiple of heads");
        }
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        const Shape& shape = input.getShape();
        if (shape.rank() != 3) {
            throw std::invalid_argument("Attention input must be (batch, length, embeddingSize), got " + shape.toString());
        }
        batch = shape[0];
        length = shape[1];
        width = shape[2];

        // The input projection packs queries, keys and values along the last axis.
        const Tensor<T> packed = inputProjection.forward(input);
        query.emplace(splitHeads(packed, 0));
        key.emplace(splitHeads(packed, 1));
        value.emplace(splitHeads(packed, 2));

        Tensor<T> scores = Ops::matmul(*query, Ops::transpose(*key, 2, 3));
        const size_t n = static_cast<size_t>(length);
        T* rowData = scores.getData().data();
        for (size_t row = 0; row < scores.getShape().size() / n; ++row, rowData += n) {
            const size_t position = row % n;
            T maxValue = -std::numeric_limits<T>::infinity();
            for (size_t k = 0; k < n; ++k) {
                rowData[k] = (causal && k > position) ? -std::numeric_limits<T>::infinity() : rowData[k] * scale();
                maxValue = std::max(maxValue, rowData[k]);
            }
            T sum = T(0);
            for (size_t k = 0; k < n; ++k) {
                rowData[k] = std::exp(rowData[k] - maxValue);
                sum += rowData[k];
            }
            for (size_t k = 0; k < n; ++k) {
                rowData[k] /= sum;
            }
        }
        weights.emplace(std::move(scores));

        Tensor<T> merged(Shape({batch, length, width}), T(0));
        mergeHeads(Ops::matmul(*weights, *value), merged, 0);
        return outputProjection.forward(merged);
    }

    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        if (!weights) {
            throw std::runtime_error("Attention backward needs a forward pass first");
        }

        const Tensor<T> gradContext = splitHeads(outputProjection.backward(gradOutput), 0);
        const Tensor<T> gradValue = Ops::matmul(Ops::transpose(*weights, 2, 3), gradContext);
        Tensor<T> gradScores = Ops::matmul(gradContext, Ops::transpose(*value, 2, 3));

        // Softmax backward per row, dScore = w * (dw - sum(dw * w)), times the score scale; masked weights are zero.
        const size_t n = static_cast<size_t>(length);
        const T* weightData = weights->getData().data();
        T* gradData = gradScores.getData().data();
        for (size_t start = 0; start < gradScores.getShape().size(); start += n) {
            T dot = T(0);
            for (size_t k = 0; k < n; ++k) {
                dot += gradData[start + k] * weightData[start + k];
            }
            for (size_t k = 0; k < n; ++k) {
                gradData[start + k] = weightData[start + k] * (gradData[start + k] - dot) * scale();
            }
        }

        Tensor<T> gradPacked(Shape({batch, length, 3 * width}), T(0));
        mergeHeads(Ops::matmul(gradScores, *key), gradPacked, 0);
        mergeHeads(Ops::matmul(Ops::transpose(gradScores, 2, 3), *query), gradPacked, 1);
        mergeHeads(gradValue, gradPacked, 2);
        return inputProjection.backward(gradPacked);
    }

    std::vector<Parameter<T>> parameters() override {
        std::vector<Parameter<T>> all = inputProjection.parameters();
        for (const Parameter<T>& parameter : outputProjection.parameters()) {
            all.push_back(parameter);
        }
        return all;
    }

    void save(std::ostream& os) const override {
        inputProjection.save(os);
        outputProjection.save(os);
    }

    void load(std::istream& is) override {
        inputProjection.load(is);
        outputProjection.load(is);
    }

private:
    int heads;
    bool causal;
    FullyConnectedLayer<T> inputProjection;
    FullyConnectedLayer<T> outputProjection;
    int batch = 0;
    int length = 0;
    int width = 0;
    std::optional<Tensor<T>> query;
    std::optional<Tensor<T>> key;
    std::optional<Tensor<T>> value;
    std::optional<Tensor<T>> weights;

    T scale() const {
        return T(1) / std::sqrt(T(width / heads));
    }

    // Copies slot `part` of (batch, length, parts * width) into (batch, heads, length, width / heads).
    Tensor<T> splitHeads(const Tensor<T>& packed, int part) const {
        const size_t headWidth = static_cast<size_t>(width / heads);
        const size_t parts = static_cast<size_t>(packed.getShape()[2] / width);
        Tensor<T> split(Shape({batch, heads, length, width / heads}), T(0));
        const T* from = packed.getData().data();
        T* to = split.getData().data();
        for (size_t b = 0; b < size_t(batch); ++b) {
            for (size_t l = 0; l < size_t(length); ++l) {
                for (size_t h = 0; h < size_t(heads); ++h) {
                    const T* row = from + ((b * length + l) * parts + part) * width + h * headWidth;
                    std::copy(row, row + headWidth, to + ((b * heads + h) * length + l) * headWidth);
                }
            }
        }
        return split;
    }

    // Writes (batch, heads, length, width / heads) into slot `part` of (batch, length, parts * width).
    void mergeHeads(const Tensor<T>& split, Tensor<T>& packed, int part) const {
        const size_t headWidth = static_cast<size_t>(width / heads);
        const size_t parts = static_cast<size_t>(packed.getShape()[2] / width);
        const T* from = split.getData().data();
        T* to = packed.getData().data();
        for (size_t b = 0; b < size_t(batch); ++b) {
            for (size_t l = 0; l < size_t(length); ++l) {
                for (size_t h = 0; h < size_t(heads); ++h) {
                    const T* row = from + ((b * heads + h) * length + l) * headWidth;
                    std::copy(row, row + headWidth, to + ((b * length + l) * parts + part) * width + h * headWidth);
                }
            }
        }
    }
};

} // namespace smart_dnn

#endif // MULTI_HEAD_ATTENTION_LAYER_HPP

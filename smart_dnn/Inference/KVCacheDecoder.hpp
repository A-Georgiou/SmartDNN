#ifndef KV_CACHE_DECODER_HPP
#define KV_CACHE_DECODER_HPP

#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Activations/GELU.hpp"
#include "smart_dnn/Layers/EmbeddingLayer.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/PositionalEmbeddingLayer.hpp"
#include "smart_dnn/Layers/TransformerBlock.hpp"
#include "smart_dnn/Regularisation/LayerNormalizationLayer.hpp"
#include "smart_dnn/Tensor/AdvancedTensorOperations.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace smart_dnn {

// Runs a trained GPT-style model one token at a time, caching each block's attention keys and values so a step only processes the new token.
template <typename T = float>
class KVCacheDecoder {
    using Ops = AdvancedTensorOperations<T>;

    struct Block {
        std::vector<Parameter<T>> parameters;
        std::vector<T> keys;
        std::vector<T> values;
    };

public:
    // The model must be Embedding, PositionalEmbedding, causal TransformerBlocks, LayerNormalization and FullyConnected layers in that order.
    KVCacheDecoder(SmartDNN<T>& model, int heads) : heads(static_cast<size_t>(heads)) {
        const size_t count = model.getLayerCount();
        if (count >= 5) {
            embedding = dynamic_cast<EmbeddingLayer<T>*>(model.getLayer(0));
            positions = dynamic_cast<PositionalEmbeddingLayer<T>*>(model.getLayer(1));
            finalNorm = dynamic_cast<LayerNormalizationLayer<T>*>(model.getLayer(count - 2));
            head = dynamic_cast<FullyConnectedLayer<T>*>(model.getLayer(count - 1));
        }
        for (size_t i = 2; i + 2 < count; ++i) {
            TransformerBlock<T>* block = dynamic_cast<TransformerBlock<T>*>(model.getLayer(i));
            if (block == nullptr) {
                break;
            }
            blocks.push_back({block->parameters(), {}, {}});
        }
        if (!embedding || !positions || !finalNorm || !head || count < 5 || blocks.size() != count - 4) {
            throw std::invalid_argument("KVCacheDecoder needs Embedding, PositionalEmbedding, TransformerBlock..., LayerNormalization and FullyConnected layers");
        }

        const Shape& table = positions->parameters()[0].value.get().getShape();
        maxLength = static_cast<size_t>(table[0]);
        width = static_cast<size_t>(table[1]);
        if (heads <= 0 || width % this->heads != 0) {
            throw std::invalid_argument("KVCacheDecoder heads must divide the embedding size");
        }
        for (Block& block : blocks) {
            block.keys.resize(maxLength * width);
            block.values.resize(maxLength * width);
        }
        norm.emplace(static_cast<int>(width));
    }

    // Appends a token and returns the logits for the token after it.
    std::vector<T> push(int token) {
        if (length == maxLength) {
            throw std::length_error("KVCacheDecoder is full; reset it and push the most recent tokens again");
        }

        Tensor<T> x = embedding->forward(Tensor<T>(Shape({1}), static_cast<T>(token)));
        const T* position = positions->parameters()[0].value.get().getData().data() + length * width;
        for (size_t i = 0; i < width; ++i) {
            x[i] += position[i];
        }
        for (Block& block : blocks) {
            const std::vector<Parameter<T>>& p = block.parameters;
            x = x + attend(block, normalize(x, p[0], p[1]));
            x = x + affine(GELU<T>().forward(affine(normalize(x, p[6], p[7]), p[8], p[9])), p[10], p[11]);
        }
        ++length;

        const Tensor<T> logits = head->forward(finalNorm->forward(x));
        return std::vector<T>(logits.getData().data(), logits.getData().data() + logits.getShape().size());
    }

    void reset() {
        length = 0;
    }

    size_t size() const {
        return length;
    }

    size_t capacity() const {
        return maxLength;
    }

private:
    size_t heads;
    size_t width = 0;
    size_t maxLength = 0;
    size_t length = 0;
    EmbeddingLayer<T>* embedding = nullptr;
    PositionalEmbeddingLayer<T>* positions = nullptr;
    LayerNormalizationLayer<T>* finalNorm = nullptr;
    FullyConnectedLayer<T>* head = nullptr;
    std::vector<Block> blocks;
    std::optional<LayerNormalizationLayer<T>> norm;

    // x times a weight matrix plus a bias row, read from a block's parameters.
    static Tensor<T> affine(const Tensor<T>& x, const Parameter<T>& weights, const Parameter<T>& bias) {
        return Ops::matmul(x, weights.value.get()) + bias.value.get();
    }

    // Applies LayerNormalization with a block norm's scale and shift, borrowing the layer's math through a scratch layer.
    Tensor<T> normalize(const Tensor<T>& x, const Parameter<T>& scale, const Parameter<T>& shift) {
        norm->parameters()[0].value.get() = scale.value.get();
        norm->parameters()[1].value.get() = shift.value.get();
        return norm->forward(x);
    }

    // Caches the new token's key and value, then attends from its query to every cached position.
    Tensor<T> attend(Block& block, const Tensor<T>& normed) {
        const Tensor<T> packed = affine(normed, block.parameters[2], block.parameters[3]);
        const T* qkv = packed.getData().data();
        std::copy(qkv + width, qkv + 2 * width, block.keys.begin() + length * width);
        std::copy(qkv + 2 * width, qkv + 3 * width, block.values.begin() + length * width);

        const size_t headWidth = width / heads;
        const T scale = T(1) / std::sqrt(static_cast<T>(headWidth));
        Tensor<T> context(Shape({1, static_cast<int>(width)}), T(0));
        std::vector<T> weights(length + 1);
        for (size_t h = 0; h < heads; ++h) {
            const size_t offset = h * headWidth;
            T maxScore = -std::numeric_limits<T>::infinity();
            for (size_t p = 0; p <= length; ++p) {
                T score = T(0);
                for (size_t i = 0; i < headWidth; ++i) {
                    score += qkv[offset + i] * block.keys[p * width + offset + i];
                }
                weights[p] = score * scale;
                maxScore = std::max(maxScore, weights[p]);
            }
            T sum = T(0);
            for (size_t p = 0; p <= length; ++p) {
                weights[p] = std::exp(weights[p] - maxScore);
                sum += weights[p];
            }
            for (size_t p = 0; p <= length; ++p) {
                for (size_t i = 0; i < headWidth; ++i) {
                    context[offset + i] += weights[p] / sum * block.values[p * width + offset + i];
                }
            }
        }
        return affine(context, block.parameters[4], block.parameters[5]);
    }
};

} // namespace smart_dnn

#endif // KV_CACHE_DECODER_HPP

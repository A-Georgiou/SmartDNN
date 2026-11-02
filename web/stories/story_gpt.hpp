#ifndef STORY_GPT_HPP
#define STORY_GPT_HPP

#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/EmbeddingLayer.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/PositionalEmbeddingLayer.hpp"
#include "smart_dnn/Layers/TransformerBlock.hpp"
#include "smart_dnn/Regularisation/LayerNormalizationLayer.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace story_gpt {

constexpr int VOCABULARY = 2048;
constexpr int CONTEXT = 256;
constexpr int WIDTH = 256;
constexpr int HEADS = 8;
constexpr int LAYERS = 6;
constexpr int STORY_END = 0x1e;

inline void addLayers(smart_dnn::SmartDNN<float>& model) {
    using namespace smart_dnn;
    model.addLayer(EmbeddingLayer<float>(VOCABULARY, WIDTH));
    model.addLayer(PositionalEmbeddingLayer<float>(CONTEXT, WIDTH));
    for (int i = 0; i < LAYERS; ++i) {
        model.addLayer(TransformerBlock<float>(WIDTH, HEADS));
    }
    model.addLayer(LayerNormalizationLayer<float>(WIDTH));
    model.addLayer(FullyConnectedLayer<float>(WIDTH, VOCABULARY));
}

// Samples from softmax(logits / temperature) restricted to the topK most likely tokens.
inline int sampleToken(const std::vector<float>& logits, float temperature, int topK, std::mt19937& rng) {
    std::vector<int> order(logits.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<int>(i);
    }
    const size_t kept = std::min(order.size(), static_cast<size_t>(std::max(1, topK)));
    std::partial_sort(order.begin(), order.begin() + static_cast<long>(kept), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    std::vector<double> weights(kept);
    for (size_t i = 0; i < kept; ++i) {
        weights[i] = std::exp((logits[order[i]] - logits[order[0]]) / temperature);
    }
    return order[std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng)];
}

} // namespace story_gpt

#endif // STORY_GPT_HPP

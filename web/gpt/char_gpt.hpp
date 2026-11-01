// The examples/CharGPT.cpp network, which trained model.bin, and its next-character sampler.
#ifndef CHAR_GPT_HPP
#define CHAR_GPT_HPP

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

constexpr int CHAR_GPT_CONTEXT = 64;

inline void addCharGptLayers(smart_dnn::SmartDNN<float>& model, int vocabularySize) {
    using namespace smart_dnn;
    constexpr int width = 128;
    model.addLayer(EmbeddingLayer<float>(vocabularySize, width));
    model.addLayer(PositionalEmbeddingLayer<float>(CHAR_GPT_CONTEXT, width));
    for (int i = 0; i < 4; ++i) {
        model.addLayer(TransformerBlock<float>(width, 4));
    }
    model.addLayer(LayerNormalizationLayer<float>(width));
    model.addLayer(FullyConnectedLayer<float>(width, vocabularySize));
}

// Samples the next id given the last CHAR_GPT_CONTEXT ids; temperatures below 1 make likely characters likelier.
inline int sampleNextCharacter(smart_dnn::SmartDNN<float>& model, const std::vector<int>& ids, int vocabularySize, float temperature, std::mt19937& rng) {
    const int length = std::min(static_cast<int>(ids.size()), CHAR_GPT_CONTEXT);
    smart_dnn::Tensor<float> context(Shape({1, length}), 0.0f);
    for (int t = 0; t < length; ++t) {
        context[t] = static_cast<float>(ids[ids.size() - length + t]);
    }
    const smart_dnn::Tensor<float> logits = model.predict(context);
    const float* last = logits.getData().data() + static_cast<size_t>(length - 1) * vocabularySize;
    const float maxLogit = *std::max_element(last, last + vocabularySize);
    std::vector<double> weights(vocabularySize);
    for (int v = 0; v < vocabularySize; ++v) {
        weights[v] = std::exp((last[v] - maxLogit) / temperature);
    }
    return std::discrete_distribution<int>(weights.begin(), weights.end())(rng);
}

#endif // CHAR_GPT_HPP

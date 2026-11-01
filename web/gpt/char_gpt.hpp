// The examples/CharGPT.cpp network, which trained model.bin, plus helpers to score it and sample from it.
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

// Logits for the character after ids[0, end), read from at most the last CHAR_GPT_CONTEXT of them.
inline std::vector<float> nextCharacterLogits(smart_dnn::SmartDNN<float>& model, const std::vector<int>& ids, size_t end, int vocabularySize) {
    const int length = static_cast<int>(std::min<size_t>(end, CHAR_GPT_CONTEXT));
    smart_dnn::Tensor<float> context(Shape({1, length}), 0.0f);
    for (int t = 0; t < length; ++t) {
        context[t] = static_cast<float>(ids[end - length + t]);
    }
    const smart_dnn::Tensor<float> logits = model.predict(context);
    const float* last = logits.getData().data() + static_cast<size_t>(length - 1) * vocabularySize;
    return std::vector<float>(last, last + vocabularySize);
}

// Samples an id from next-character logits; temperatures below 1 make likely characters likelier.
inline int sampleCharacter(const std::vector<float>& logits, float temperature, std::mt19937& rng) {
    const float maxLogit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> weights(logits.size());
    for (size_t v = 0; v < logits.size(); ++v) {
        weights[v] = std::exp((logits[v] - maxLogit) / temperature);
    }
    return std::discrete_distribution<int>(weights.begin(), weights.end())(rng);
}

#endif // CHAR_GPT_HPP

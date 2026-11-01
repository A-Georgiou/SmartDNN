// Trains a small character-level GPT on a text file, then samples: c++ -std=c++17 -O3 -I . examples/CharGPT.cpp -o chargpt && ./chargpt input.txt [steps]
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Datasets/CharacterDataset.hpp"
#include "smart_dnn/Layers/EmbeddingLayer.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/PositionalEmbeddingLayer.hpp"
#include "smart_dnn/Layers/TransformerBlock.hpp"
#include "smart_dnn/Loss/SoftmaxCrossEntropyLoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "smart_dnn/Regularisation/LayerNormalizationLayer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

using namespace smart_dnn;

constexpr int LENGTH = 64;
constexpr int WIDTH = 128;
constexpr int HEADS = 4;
constexpr int LAYERS = 4;
constexpr int BATCH = 32;
constexpr float PEAK_RATE = 1e-3f;
constexpr int WARMUP = 100;

// Linear warmup, then cosine decay to a tenth of the peak.
float learningRate(int step, int steps) {
    if (step < WARMUP) {
        return PEAK_RATE * static_cast<float>(step + 1) / WARMUP;
    }
    const float progress = static_cast<float>(step - WARMUP) / static_cast<float>(std::max(1, steps - WARMUP));
    return PEAK_RATE * (0.1f + 0.45f * (1.0f + std::cos(3.14159265f * progress)));
}

// Samples one character at a time from the model's next-character distribution.
std::string generate(SmartDNN<float>& model, const CharacterDataset<float>& data, const std::string& prompt, int count, float temperature, std::mt19937& rng) {
    std::vector<int> ids = data.encode(prompt);
    const int vocabulary = data.vocabularySize();
    for (int i = 0; i < count; ++i) {
        const int length = std::min(static_cast<int>(ids.size()), LENGTH);
        Tensor<float> context(Shape({1, length}), 0.0f);
        for (int t = 0; t < length; ++t) {
            context[t] = static_cast<float>(ids[ids.size() - length + t]);
        }
        const Tensor<float> logits = model.predict(context);
        const float* last = logits.getData().data() + static_cast<size_t>(length - 1) * vocabulary;
        const float maxLogit = *std::max_element(last, last + vocabulary);
        std::vector<double> weights(vocabulary);
        for (int v = 0; v < vocabulary; ++v) {
            weights[v] = std::exp((last[v] - maxLogit) / temperature);
        }
        ids.push_back(std::discrete_distribution<int>(weights.begin(), weights.end())(rng));
    }
    return data.decode(ids);
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : ".datasets/tinyshakespeare.txt";
    const int steps = argc > 2 ? std::atoi(argv[2]) : 3000;
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Cannot read " << path << ". Tiny Shakespeare: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt\n";
        return 1;
    }
    std::stringstream text;
    text << file.rdbuf();
    const CharacterDataset<float> data(text.str());
    const size_t split = data.getTokens().size() * 9 / 10;

    RandomEngine::setSeed(1337);
    std::mt19937 rng(1337);
    SmartDNN<float> model;
    model.addLayer(EmbeddingLayer<float>(data.vocabularySize(), WIDTH));
    model.addLayer(PositionalEmbeddingLayer<float>(LENGTH, WIDTH));
    for (int i = 0; i < LAYERS; ++i) {
        model.addLayer(TransformerBlock<float>(WIDTH, HEADS));
    }
    model.addLayer(LayerNormalizationLayer<float>(WIDTH));
    model.addLayer(FullyConnectedLayer<float>(WIDTH, data.vocabularySize()));
    model.compile(SoftmaxCrossEntropyLoss<float>(), AdamOptimizer<float>());
    model.setMaxGradientNorm(1.0f);

    SoftmaxCrossEntropyLoss<float> loss;
    const auto start = std::chrono::steady_clock::now();
    float trainLoss = 0.0f;
    int trainSteps = 0;
    for (int step = 1; step <= steps; ++step) {
        auto [inputs, targets] = data.sampleBatch(BATCH, LENGTH, rng, 0, split);
        trainLoss += model.trainStep(inputs, targets, learningRate(step - 1, steps));
        ++trainSteps;
        if (step % 100 == 0 || step == steps) {
            float validationLoss = 0.0f;
            for (int i = 0; i < 10; ++i) {
                auto [x, y] = data.sampleBatch(BATCH, LENGTH, rng, split, data.getTokens().size());
                validationLoss += loss.compute(model.predict(x), y)[0] / 10.0f;
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf("step %5d  train %.3f  validation %.3f  %.0fs\n", step, trainLoss / trainSteps, validationLoss, seconds);
            std::fflush(stdout);
            trainLoss = 0.0f;
            trainSteps = 0;
        }
    }

    model.saveModel("chargpt.bin");
    std::cout << generate(model, data, "\n", 500, 0.8f, rng) << std::endl;
}

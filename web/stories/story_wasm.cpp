// WebAssembly entry points for the story GPT demo: load the embedded model and tokenizer, start a story from a prompt and sample tokens.
#include "story_gpt.hpp"
#include "smart_dnn/Datasets/BytePairTokenizer.hpp"
#include "smart_dnn/Inference/KVCacheDecoder.hpp"
#include <emscripten/emscripten.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace smart_dnn;

namespace {
SmartDNN<float> model;
std::optional<KVCacheDecoder<float>> decoder;
BytePairTokenizer tokenizer;
std::vector<int> context;
std::vector<float> logits;
std::string piece;
std::mt19937 rng;

// Restarts the cache from the latest tokens of the context, at most count of them, keeping the logits that follow.
void refill(size_t count) {
    decoder->reset();
    for (size_t i = context.size() - std::min(count, context.size()); i < context.size(); ++i) {
        logits = decoder->push(context[i]);
    }
}

std::vector<int> storyTokens(const char* text) {
    return tokenizer.encode(std::string(1, static_cast<char>(story_gpt::STORY_END)) + text);
}
}

extern "C" {

// Builds the network, loads /model.bin and /tokenizer.txt (embedded by build.sh) and returns the vocabulary size.
EMSCRIPTEN_KEEPALIVE int story_load() {
    std::ifstream file("/tokenizer.txt");
    tokenizer = BytePairTokenizer::load(file);
    story_gpt::addLayers(model);
    model.loadModel("/model.bin");
    model.evalMode();
    decoder.emplace(model);
    return tokenizer.vocabularySize();
}

// Starts a new story from the prompt and seeds the sampler; returns the number of prompt tokens.
EMSCRIPTEN_KEEPALIVE int story_begin(const char* prompt, unsigned seed) {
    context = storyTokens(prompt);
    refill(decoder->capacity());
    rng.seed(seed);
    return static_cast<int>(context.size()) - 1;
}

// Samples the next token through the KV cache and returns its byte length, or -1 when the story ends; story_piece points at the bytes.
EMSCRIPTEN_KEEPALIVE int story_next(float temperature, int topK) {
    const int next = story_gpt::sampleToken(logits, temperature, topK, rng);
    if (next == story_gpt::STORY_END) {
        return -1;
    }
    context.push_back(next);
    if (decoder->size() == decoder->capacity()) {
        refill(decoder->capacity() / 2);
    } else {
        logits = decoder->push(next);
    }
    piece = tokenizer.decode({next});
    return static_cast<int>(piece.size());
}

EMSCRIPTEN_KEEPALIVE const char* story_piece() {
    return piece.data();
}

// Average negative log-likelihood per token of the text as the start of a story, over its first CONTEXT tokens; this ends the current story.
EMSCRIPTEN_KEEPALIVE float story_score(const char* text) {
    const std::vector<int> ids = storyTokens(text);
    const size_t count = std::min(ids.size(), decoder->capacity());
    decoder->reset();
    double total = 0.0;
    for (size_t i = 0; i + 1 < count; ++i) {
        const std::vector<float> next = decoder->push(ids[i]);
        const float maxLogit = *std::max_element(next.begin(), next.end());
        double sum = 0.0;
        for (float logit : next) {
            sum += std::exp(logit - maxLogit);
        }
        total += std::log(sum) - (next[static_cast<size_t>(ids[i + 1])] - maxLogit);
    }
    return count > 1 ? static_cast<float>(total / static_cast<double>(count - 1)) : 0.0f;
}

}

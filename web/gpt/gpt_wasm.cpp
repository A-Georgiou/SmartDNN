// WebAssembly entry points for the Shakespeare GPT demo: load the embedded model, start from a prompt and sample characters.
#include "char_gpt.hpp"
#include "smart_dnn/Loss/SoftmaxCrossEntropyLoss.hpp"
#include <emscripten/emscripten.h>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace smart_dnn;

namespace {
SmartDNN<float> model;
std::string vocabulary;
int idOf[256];
std::vector<int> context;
std::mt19937 rng;

// Ids of the characters the model knows; any others are skipped.
std::vector<int> encode(const char* text) {
    std::vector<int> ids;
    for (const unsigned char* c = reinterpret_cast<const unsigned char*>(text); *c != 0; ++c) {
        if (idOf[*c] >= 0) {
            ids.push_back(idOf[*c]);
        }
    }
    return ids;
}
}

extern "C" {

// Builds the network, loads /model.bin and /vocab.txt (embedded by build.sh) and returns the vocabulary size.
EMSCRIPTEN_KEEPALIVE int gpt_load() {
    std::ifstream file("/vocab.txt", std::ios::binary);
    vocabulary.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    std::fill(std::begin(idOf), std::end(idOf), -1);
    for (size_t i = 0; i < vocabulary.size(); ++i) {
        idOf[static_cast<unsigned char>(vocabulary[i])] = static_cast<int>(i);
    }
    addCharGptLayers(model, static_cast<int>(vocabulary.size()));
    model.loadModel("/model.bin");
    model.evalMode();
    return static_cast<int>(vocabulary.size());
}

// Starts a new text from the prompt, or from a new line if it has no known characters, and seeds the sampler.
EMSCRIPTEN_KEEPALIVE void gpt_reset(const char* prompt, unsigned seed) {
    context = encode(prompt);
    if (context.empty()) {
        context.push_back(idOf[static_cast<unsigned char>('\n')]);
    }
    rng.seed(seed);
}

// Samples the next character at the given temperature, appends it to the context and returns its character code.
EMSCRIPTEN_KEEPALIVE int gpt_next(float temperature) {
    const int id = sampleNextCharacter(model, context, static_cast<int>(vocabulary.size()), temperature, rng);
    context.push_back(id);
    return static_cast<unsigned char>(vocabulary[id]);
}

// Average negative log-likelihood per character of the text, predicting each from up to CHAR_GPT_CONTEXT earlier ones.
EMSCRIPTEN_KEEPALIVE float gpt_score(const char* text) {
    const std::vector<int> ids = encode(text);
    SoftmaxCrossEntropyLoss<float> loss;
    float total = 0.0f;
    size_t count = 0;
    for (size_t start = 0; start + 1 < ids.size(); start += CHAR_GPT_CONTEXT) {
        const int length = static_cast<int>(std::min<size_t>(CHAR_GPT_CONTEXT, ids.size() - 1 - start));
        Tensor<float> inputs(Shape({1, length}), 0.0f);
        Tensor<float> targets(Shape({1, length}), 0.0f);
        for (int t = 0; t < length; ++t) {
            inputs[t] = static_cast<float>(ids[start + t]);
            targets[t] = static_cast<float>(ids[start + t + 1]);
        }
        total += loss.compute(model.predict(inputs), targets)[0] * static_cast<float>(length);
        count += static_cast<size_t>(length);
    }
    return count > 0 ? total / static_cast<float>(count) : 0.0f;
}

}

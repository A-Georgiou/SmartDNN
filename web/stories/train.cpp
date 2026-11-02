// Trains the story GPT on TinyStoriesV2-GPT4 (huggingface.co/datasets/roneneldan/TinyStories), writing tokenizer.txt, the best model as 8-bit model.bin and a float checkpoint.bin to resume from.
#include "story_gpt.hpp"
#include "smart_dnn/Datasets/BytePairTokenizer.hpp"
#include "smart_dnn/Inference/KVCacheDecoder.hpp"
#include "smart_dnn/Loss/SoftmaxCrossEntropyLoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "smart_dnn/Training/DataParallelTrainer.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

using namespace smart_dnn;
using Clock = std::chrono::steady_clock;

constexpr size_t TRAIN_BYTES = 600u << 20;
constexpr size_t TOKENIZER_BYTES = 100u << 20;
constexpr int ROWS_PER_THREAD = 4;
constexpr int VALID_ROWS = 64;
constexpr int WARMUP = 300;
constexpr float PEAK_RATE = 1e-3f;

// Reads up to limit bytes of a TinyStories file, replacing each <|endoftext|> line with one 0x1E byte.
std::string readStories(const std::string& path, size_t limit) {
    std::ifstream file(path, std::ios::binary);
    std::string raw(limit, '\0');
    file.read(raw.data(), static_cast<std::streamsize>(limit));
    raw.resize(static_cast<size_t>(file.gcount()));
    const std::string marker = "<|endoftext|>";
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size();) {
        if (raw.compare(i, marker.size(), marker) == 0) {
            while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) {
                out.pop_back();
            }
            out += static_cast<char>(story_gpt::STORY_END);
            for (i += marker.size(); i < raw.size() && (raw[i] == '\n' || raw[i] == ' '); ++i) {
            }
        } else {
            out += raw[i++];
        }
    }
    return out;
}

std::pair<Tensor<float>, Tensor<float>> windows(const std::vector<int>& ids, int rows, std::mt19937& rng) {
    std::uniform_int_distribution<size_t> start(0, ids.size() - story_gpt::CONTEXT - 1);
    Tensor<float> inputs(Shape({rows, story_gpt::CONTEXT}), 0.0f);
    Tensor<float> targets(Shape({rows, story_gpt::CONTEXT}), 0.0f);
    for (int r = 0; r < rows; ++r) {
        const size_t first = start(rng);
        for (int t = 0; t < story_gpt::CONTEXT; ++t) {
            inputs[static_cast<size_t>(r * story_gpt::CONTEXT + t)] = static_cast<float>(ids[first + t]);
            targets[static_cast<size_t>(r * story_gpt::CONTEXT + t)] = static_cast<float>(ids[first + t + 1]);
        }
    }
    return {inputs, targets};
}

float learningRate(int step, int steps) {
    if (step < WARMUP) {
        return PEAK_RATE * static_cast<float>(step + 1) / WARMUP;
    }
    const float progress = static_cast<float>(step - WARMUP) / static_cast<float>(std::max(1, steps - WARMUP));
    return PEAK_RATE * (0.1f + 0.45f * (1.0f + std::cos(3.14159265f * progress)));
}

float validationLoss(SmartDNN<float>& model, const Tensor<float>& inputs, const Tensor<float>& targets) {
    SoftmaxCrossEntropyLoss<float> loss;
    float total = 0.0f;
    const int chunk = 8;
    for (int first = 0; first < VALID_ROWS; first += chunk) {
        const size_t offset = static_cast<size_t>(first) * story_gpt::CONTEXT;
        const Shape shape({chunk, story_gpt::CONTEXT});
        const Tensor<float> x(shape, inputs.getData().data() + offset);
        const Tensor<float> y(shape, targets.getData().data() + offset);
        total += loss.compute(model.predict(x), y)[0] / (VALID_ROWS / chunk);
    }
    return total;
}

std::string sampleStory(SmartDNN<float>& model, const BytePairTokenizer& tokenizer, std::mt19937& rng) {
    KVCacheDecoder<float> decoder(model);
    std::vector<int> ids = tokenizer.encode(std::string(1, static_cast<char>(story_gpt::STORY_END)) + "Once upon a time");
    std::vector<float> logits;
    for (int id : ids) {
        logits = decoder.push(id);
    }
    const size_t prompt = ids.size();
    while (ids.size() - prompt < 200 && decoder.size() < decoder.capacity()) {
        const int next = story_gpt::sampleToken(logits, 0.8f, 40, rng);
        if (next == story_gpt::STORY_END) {
            break;
        }
        ids.push_back(next);
        logits = decoder.push(next);
    }
    return tokenizer.decode(std::vector<int>(ids.begin() + 1, ids.end()));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: story_train <dir with TinyStoriesV2-GPT4-train.txt and -valid.txt> <out dir, web/stories for the demo> [steps] [threads] [resume step]\n");
        return 1;
    }
    const std::string dataDir = argv[1];
    const std::string outDir = argv[2];
    const int steps = argc > 3 ? std::atoi(argv[3]) : 7000;
    const int threads = argc > 4 ? std::atoi(argv[4]) : 16;
    const int resumeStep = argc > 5 ? std::atoi(argv[5]) : 0;
    const auto start = Clock::now();
    auto seconds = [&] { return std::chrono::duration<double>(Clock::now() - start).count(); };

    const std::string train = readStories(dataDir + "/TinyStoriesV2-GPT4-train.txt", TRAIN_BYTES);
    const std::string valid = readStories(dataDir + "/TinyStoriesV2-GPT4-valid.txt", 64u << 20);
    BytePairTokenizer tokenizer;
    if (resumeStep > 0) {
        std::ifstream in(outDir + "/tokenizer.txt");
        tokenizer = BytePairTokenizer::load(in);
    } else {
        tokenizer = BytePairTokenizer::train(train.substr(0, TOKENIZER_BYTES), story_gpt::VOCABULARY);
        std::ofstream out(outDir + "/tokenizer.txt");
        tokenizer.save(out);
    }
    const std::vector<int> trainIds = tokenizer.encode(train);
    const std::vector<int> validIds = tokenizer.encode(valid);
    const double charsPerToken = static_cast<double>(valid.size()) / static_cast<double>(validIds.size());
    std::printf("data: %zu train tokens, %zu valid tokens, %.2f chars/token, vocab %d, ready in %.0fs\n", trainIds.size(), validIds.size(),
                charsPerToken, tokenizer.vocabularySize(), seconds());

    RandomEngine::setSeed(1337);
    SmartDNN<float> model;
    story_gpt::addLayers(model);
    AdamOptions<float> adam;
    adam.beta2 = 0.95f;
    model.compile(SoftmaxCrossEntropyLoss<float>(), AdamOptimizer<float>(adam));
    model.setMaxGradientNorm(1.0f);
    if (resumeStep > 0) {
        model.loadModel(outDir + "/checkpoint.bin");
    }
    DataParallelTrainer<float> trainer(model, story_gpt::addLayers, SoftmaxCrossEntropyLoss<float>(), threads);

    std::mt19937 validRng(7);
    const auto [validX, validY] = windows(validIds, VALID_ROWS, validRng);
    std::mt19937 rng(1337u + static_cast<unsigned>(resumeStep));
    float best = 1e9f;
    float trainLoss = 0.0f;
    int trainSteps = 0;
    double lastTime = seconds();
    for (int step = resumeStep + 1; step <= steps; ++step) {
        auto [x, y] = windows(trainIds, threads * ROWS_PER_THREAD, rng);
        trainLoss += trainer.trainStep(x, y, learningRate(step - 1, steps));
        ++trainSteps;
        if (step % 50 == 0 || step == steps) {
            const double now = seconds();
            const double tokensPerSecond = trainSteps * static_cast<double>(x.getShape().size()) / (now - lastTime);
            std::printf("step %5d  train %.3f  lr %.2e  %.0f tokens/s  %.0fs  eta %.0fm\n", step, trainLoss / trainSteps, learningRate(step - 1, steps),
                        tokensPerSecond, now, (steps - step) * x.getShape().size() / tokensPerSecond / 60.0);
            std::fflush(stdout);
            trainLoss = 0.0f;
            trainSteps = 0;
            lastTime = now;
        }
        if (step % 500 == 0 || step == steps) {
            const float validation = validationLoss(model, validX, validY);
            model.saveModel(outDir + "/checkpoint.bin");
            if (validation < best) {
                best = validation;
                model.saveModel(outDir + "/model.bin", WeightFormat::Int8);
            }
            std::printf("step %5d  validation %.4f (%.4f nats/char, best %.4f)\n  sample: %s\n", step, validation, validation / charsPerToken, best,
                        sampleStory(model, tokenizer, rng).c_str());
            std::fflush(stdout);
            lastTime = seconds();
        }
    }
}

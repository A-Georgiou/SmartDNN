#ifndef CHARACTER_DATASET_HPP
#define CHARACTER_DATASET_HPP

#include "smart_dnn/Tensor/Tensor.hpp"
#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace smart_dnn {

// Character-level text for language models: one id per distinct character, and random next-character training windows.
template <typename T = float>
class CharacterDataset {
public:
    explicit CharacterDataset(const std::string& text) {
        idOf.fill(-1);
        for (unsigned char c : text) {
            idOf[c] = 0;
        }
        for (size_t c = 0; c < idOf.size(); ++c) {
            if (idOf[c] == 0) {
                idOf[c] = static_cast<int>(characters.size());
                characters.push_back(static_cast<char>(c));
            }
        }
        tokens = encode(text);
        if (tokens.size() < 2) {
            throw std::invalid_argument("CharacterDataset needs at least two characters of text");
        }
    }

    int vocabularySize() const {
        return static_cast<int>(characters.size());
    }

    const std::vector<int>& getTokens() const {
        return tokens;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (unsigned char c : text) {
            if (idOf[c] < 0) {
                throw std::invalid_argument("Character is not in the dataset's vocabulary");
            }
            ids.push_back(idOf[c]);
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string text;
        for (int id : ids) {
            text += characters.at(static_cast<size_t>(id));
        }
        return text;
    }

    // Random windows from tokens [begin, end): inputs (batch, length) and, as targets, the same windows one character later.
    std::pair<Tensor<T>, Tensor<T>> sampleBatch(int batch, int length, std::mt19937& rng, size_t begin, size_t end) const {
        end = std::min(end, tokens.size());
        if (batch <= 0 || length <= 0 || end < begin + static_cast<size_t>(length) + 1) {
            throw std::invalid_argument("sampleBatch needs a range longer than the window length");
        }

        std::uniform_int_distribution<size_t> startDistribution(begin, end - static_cast<size_t>(length) - 1);
        Tensor<T> inputs(Shape({batch, length}), T(0));
        Tensor<T> targets(Shape({batch, length}), T(0));
        for (size_t b = 0; b < static_cast<size_t>(batch); ++b) {
            const size_t start = startDistribution(rng);
            for (size_t t = 0; t < static_cast<size_t>(length); ++t) {
                inputs[b * length + t] = T(tokens[start + t]);
                targets[b * length + t] = T(tokens[start + t + 1]);
            }
        }
        return {inputs, targets};
    }

private:
    std::array<int, 256> idOf;
    std::vector<char> characters;
    std::vector<int> tokens;
};

} // namespace smart_dnn

#endif // CHARACTER_DATASET_HPP

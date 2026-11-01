#ifndef BYTE_PAIR_TOKENIZER_HPP
#define BYTE_PAIR_TOKENIZER_HPP

#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace smart_dnn {

// Byte-level BPE: ids 0-255 are bytes, each later id joins two earlier ones, and merges never cross words, whitespace or control bytes.
class BytePairTokenizer {
public:
    // Learns merges from text until there are vocabularySize ids or no pair occurs twice.
    static BytePairTokenizer train(const std::string& text, int vocabularySize) {
        if (vocabularySize < 256) {
            throw std::invalid_argument("BytePairTokenizer needs at least the 256 byte ids");
        }
        std::unordered_map<std::string_view, int64_t> chunkCounts;
        forEachChunk(text, [&](std::string_view chunk) { ++chunkCounts[chunk]; });
        std::vector<std::vector<int>> words;
        std::vector<int64_t> weights;
        for (const auto& [chunk, count] : chunkCounts) {
            words.emplace_back(chunk.begin(), chunk.end());
            for (int& symbol : words.back()) {
                symbol = static_cast<unsigned char>(symbol);
            }
            weights.push_back(count);
        }

        std::unordered_map<uint64_t, int64_t> pairCounts;
        auto countPairs = [&](const std::vector<int>& word, int64_t weight) {
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                const auto found = pairCounts.emplace(pairKey(word[i], word[i + 1]), 0).first;
                if ((found->second += weight) == 0) {
                    pairCounts.erase(found);
                }
            }
        };
        for (size_t w = 0; w < words.size(); ++w) {
            countPairs(words[w], weights[w]);
        }

        BytePairTokenizer tokenizer;
        while (tokenizer.vocabularySize() < vocabularySize) {
            auto best = pairCounts.end();
            for (auto it = pairCounts.begin(); it != pairCounts.end(); ++it) {
                if (best == pairCounts.end() || it->second > best->second || (it->second == best->second && it->first < best->first)) {
                    best = it;
                }
            }
            if (best == pairCounts.end() || best->second < 2) {
                break;
            }
            const int first = static_cast<int>(best->first >> 32);
            const int second = static_cast<int>(best->first & 0xffffffffu);
            const int merged = tokenizer.vocabularySize();
            tokenizer.addMerge(first, second);
            for (size_t w = 0; w < words.size(); ++w) {
                if (contains(words[w], first, second)) {
                    countPairs(words[w], -weights[w]);
                    mergeAll(words[w], first, second, merged);
                    countPairs(words[w], weights[w]);
                }
            }
        }
        return tokenizer;
    }

    BytePairTokenizer() {
        for (int byte = 0; byte < 256; ++byte) {
            pieces.emplace_back(1, static_cast<char>(byte));
        }
    }

    int vocabularySize() const {
        return static_cast<int>(pieces.size());
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        std::unordered_map<std::string_view, std::vector<int>> cache;
        forEachChunk(text, [&](std::string_view chunk) {
            auto found = cache.find(chunk);
            if (found == cache.end()) {
                found = cache.emplace(chunk, encodeChunk(chunk)).first;
            }
            ids.insert(ids.end(), found->second.begin(), found->second.end());
        });
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string text;
        for (int id : ids) {
            if (id < 0 || id >= vocabularySize()) {
                throw std::out_of_range("Token id " + std::to_string(id) + " is outside the vocabulary");
            }
            text += pieces[static_cast<size_t>(id)];
        }
        return text;
    }

    void save(std::ostream& os) const {
        os << "bpe " << merges.size() << '\n';
        for (const auto& [first, second] : merges) {
            os << first << ' ' << second << '\n';
        }
    }

    static BytePairTokenizer load(std::istream& is) {
        std::string magic;
        size_t count = 0;
        if (!(is >> magic >> count) || magic != "bpe") {
            throw std::runtime_error("Not a BytePairTokenizer file");
        }
        BytePairTokenizer tokenizer;
        for (size_t i = 0; i < count; ++i) {
            int first = -1;
            int second = -1;
            if (!(is >> first >> second) || first < 0 || second < 0 || first >= tokenizer.vocabularySize() || second >= tokenizer.vocabularySize()) {
                throw std::runtime_error("BytePairTokenizer merge " + std::to_string(i) + " is missing or refers to an unknown id");
            }
            tokenizer.addMerge(first, second);
        }
        return tokenizer;
    }

private:
    enum class Kind { Letter, Digit, Space, Control, Other };

    struct Character {
        Kind kind;
        size_t length;
    };

    static Kind kindOf(uint32_t code) {
        if ((code >= 'a' && code <= 'z') || (code >= 'A' && code <= 'Z')) {
            return Kind::Letter;
        }
        if (code >= '0' && code <= '9') {
            return Kind::Digit;
        }
        if (code == ' ' || code == '\n' || code == '\t' || code == '\r' || code == 0xa0 || code == 0x1680 || (code >= 0x2000 && code <= 0x200a)
            || code == 0x2028 || code == 0x2029 || code == 0x202f || code == 0x205f || code == 0x3000) {
            return Kind::Space;
        }
        // Control bytes such as 0x1E stay single tokens, so they can mark document ends.
        if (code < 0x20 || code == 0x7f) {
            return Kind::Control;
        }
        // ASCII and Latin-1 symbols, general punctuation through symbols and arrows, CJK and full-width punctuation, emoji.
        const bool other = code < 0x80 || (code >= 0xa1 && code <= 0xbf) || code == 0xd7 || code == 0xf7 || (code >= 0x200b && code <= 0x2bff)
            || (code >= 0x3001 && code <= 0x303f) || (code >= 0xff01 && code <= 0xff0f) || (code >= 0x1f000 && code <= 0x1faff);
        return other ? Kind::Other : Kind::Letter;
    }

    // Decodes the UTF-8 character at text[i]; a malformed byte counts as one punctuation byte.
    static Character characterAt(const std::string& text, size_t i) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        const size_t length = lead < 0x80 ? 1 : lead >= 0xf0 ? 4 : lead >= 0xe0 ? 3 : lead >= 0xc0 ? 2 : 0;
        if (length == 0 || i + length > text.size()) {
            return {Kind::Other, 1};
        }
        uint32_t code = length == 1 ? lead : lead & (0x7fu >> length);
        for (size_t k = 1; k < length; ++k) {
            const unsigned char next = static_cast<unsigned char>(text[i + k]);
            if ((next & 0xc0) != 0x80) {
                return {Kind::Other, 1};
            }
            code = (code << 6) | (next & 0x3fu);
        }
        return {kindOf(code), length};
    }

    static bool takesLeadingSpace(Kind kind) {
        return kind == Kind::Letter || kind == Kind::Digit || kind == Kind::Other;
    }

    // Length of an apostrophe (' or U+2019) at text[i] that sits between letters, otherwise 0.
    static size_t innerApostrophe(const std::string& text, size_t i) {
        const size_t length = text[i] == '\'' ? 1 : text.compare(i, 3, "\xe2\x80\x99") == 0 ? 3 : 0;
        return length > 0 && i + length < text.size() && characterAt(text, i + length).kind == Kind::Letter ? length : 0;
    }

    // Splits text like GPT-2: a word, number or punctuation run with at most one leading space, a whitespace run, or a control byte.
    template <typename Visit>
    static void forEachChunk(const std::string& text, Visit visit) {
        const size_t size = text.size();
        size_t i = 0;
        while (i < size) {
            const size_t start = i;
            Kind kind = characterAt(text, i).kind;
            if (text[i] == ' ' && i + 1 < size && takesLeadingSpace(characterAt(text, i + 1).kind)) {
                kind = characterAt(text, ++i).kind;
            }
            if (kind == Kind::Control) {
                ++i;
            } else if (kind == Kind::Space) {
                size_t last = i;
                for (Character next = characterAt(text, i); next.kind == Kind::Space; next = characterAt(text, i)) {
                    last = i;
                    if ((i += next.length) == size) {
                        break;
                    }
                }
                // Leaves the last space of a longer run to the word after it.
                if (i < size && last > start && text[last] == ' ' && takesLeadingSpace(characterAt(text, i).kind)) {
                    i = last;
                }
            } else {
                while (i < size) {
                    const Character next = characterAt(text, i);
                    const size_t apostrophe = kind == Kind::Letter && next.kind != kind ? innerApostrophe(text, i) : 0;
                    if (next.kind != kind && apostrophe == 0) {
                        break;
                    }
                    i += apostrophe > 0 ? apostrophe : next.length;
                }
            }
            visit(std::string_view(text.data() + start, i - start));
        }
    }

    static uint64_t pairKey(int first, int second) {
        return (static_cast<uint64_t>(first) << 32) | static_cast<uint32_t>(second);
    }

    static bool contains(const std::vector<int>& word, int first, int second) {
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            if (word[i] == first && word[i + 1] == second) {
                return true;
            }
        }
        return false;
    }

    static void mergeAll(std::vector<int>& word, int first, int second, int merged) {
        size_t out = 0;
        for (size_t i = 0; i < word.size(); ++i) {
            if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                word[out++] = merged;
                ++i;
            } else {
                word[out++] = word[i];
            }
        }
        word.resize(out);
    }

    void addMerge(int first, int second) {
        mergedIds[pairKey(first, second)] = vocabularySize();
        merges.emplace_back(first, second);
        pieces.push_back(pieces[static_cast<size_t>(first)] + pieces[static_cast<size_t>(second)]);
    }

    // Applies merges in the order they were learned, which reproduces training on this chunk.
    std::vector<int> encodeChunk(std::string_view chunk) const {
        std::vector<int> symbols(chunk.begin(), chunk.end());
        for (int& symbol : symbols) {
            symbol = static_cast<unsigned char>(symbol);
        }
        while (symbols.size() > 1) {
            int best = std::numeric_limits<int>::max();
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto found = mergedIds.find(pairKey(symbols[i], symbols[i + 1]));
                if (found != mergedIds.end() && found->second < best) {
                    best = found->second;
                }
            }
            if (best == std::numeric_limits<int>::max()) {
                break;
            }
            const auto [first, second] = merges[static_cast<size_t>(best - 256)];
            mergeAll(symbols, first, second, best);
        }
        return symbols;
    }

    std::vector<std::pair<int, int>> merges;
    std::unordered_map<uint64_t, int> mergedIds;
    std::vector<std::string> pieces;
};

} // namespace smart_dnn

#endif // BYTE_PAIR_TOKENIZER_HPP

#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>
#include "../../smart_dnn/Datasets/BytePairTokenizer.hpp"

namespace smart_dnn {

TEST(BytePairTokenizerTest, RoundTripsAnyBytes) {
    const BytePairTokenizer tokenizer = BytePairTokenizer::train("the cat sat on the mat. the cat ran.", 280);
    const std::string text = "The cat's 42 mats,\n\n  na\xc3\xafve \x1e\x01 caf\xc3\xa9!";
    EXPECT_EQ(tokenizer.decode(tokenizer.encode(text)), text);
}

TEST(BytePairTokenizerTest, MergesFrequentWordsButNotAcrossBoundaries) {
    const BytePairTokenizer tokenizer = BytePairTokenizer::train(" cat cat cat. cat\x1e\x1e", 300);
    EXPECT_EQ(tokenizer.encode(" cat").size(), 1u);
    EXPECT_EQ(tokenizer.encode(" cat.").size(), 2u);
    EXPECT_EQ(tokenizer.encode("\x1e\x1e"), (std::vector<int>{0x1e, 0x1e}));
    EXPECT_EQ(tokenizer.vocabularySize(), 259);
}

TEST(BytePairTokenizerTest, KeepsUnicodePunctuationAndSpacesOutOfWords) {
    std::string text;
    for (int i = 0; i < 3; ++i) {
        text += "\xe2\x80\x9cHi\xe2\x80\x9d said Sam.\xc2\xa0";
    }
    const BytePairTokenizer tokenizer = BytePairTokenizer::train(text, 300);
    auto pieces = [&](const std::string& input) {
        std::vector<std::string> out;
        for (int id : tokenizer.encode(input)) {
            out.push_back(tokenizer.decode({id}));
        }
        return out;
    };
    EXPECT_EQ(pieces("\xe2\x80\x9cHi\xe2\x80\x9d"), (std::vector<std::string>{"\xe2\x80\x9c", "Hi", "\xe2\x80\x9d"}));
    EXPECT_EQ(pieces(" Sam.\xc2\xa0"), (std::vector<std::string>{" Sam", ".", "\xc2\xa0"}));
}

TEST(BytePairTokenizerTest, SavesAndLoadsTheSameMerges) {
    const BytePairTokenizer tokenizer = BytePairTokenizer::train("low lower lowest newer newest", 270);
    std::stringstream stream;
    tokenizer.save(stream);
    const BytePairTokenizer loaded = BytePairTokenizer::load(stream);
    EXPECT_EQ(loaded.vocabularySize(), tokenizer.vocabularySize());
    EXPECT_EQ(loaded.encode(" lowest newer"), tokenizer.encode(" lowest newer"));

    std::stringstream unknownId("bpe 1\n300 2\n");
    EXPECT_THROW(BytePairTokenizer::load(unknownId), std::runtime_error);
}

} // namespace smart_dnn

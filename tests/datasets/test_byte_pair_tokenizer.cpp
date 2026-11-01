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

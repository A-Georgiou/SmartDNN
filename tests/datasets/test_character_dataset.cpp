#ifndef TEST_CHARACTER_DATASET_CPP
#define TEST_CHARACTER_DATASET_CPP

#include <gtest/gtest.h>
#include <random>
#include "../../smart_dnn/Datasets/CharacterDataset.hpp"

namespace smart_dnn {

TEST(CharacterDatasetTest, EncodesAndDecodesItsCharacters) {
    CharacterDataset<float> data("hello world");

    EXPECT_EQ(data.vocabularySize(), 8);
    EXPECT_EQ(data.decode(data.encode("lower")), "lower");
    EXPECT_THROW(data.encode("hi"), std::invalid_argument);
}

TEST(CharacterDatasetTest, SamplesWindowsWithTheirNextCharacters) {
    // Every character is distinct, so ids count up along the text.
    CharacterDataset<float> data("abcdefghij");
    std::mt19937 rng(3);

    auto [inputs, targets] = data.sampleBatch(16, 3, rng, 2, 8);

    ASSERT_EQ(inputs.getShape(), Shape({16, 3}));
    for (size_t b = 0; b < 16; ++b) {
        EXPECT_GE(inputs[b * 3], 2.0f);
        EXPECT_LE(inputs[b * 3], 4.0f);
        for (size_t t = 0; t < 3; ++t) {
            EXPECT_EQ(inputs[b * 3 + t], inputs[b * 3] + static_cast<float>(t));
            EXPECT_EQ(targets[b * 3 + t], inputs[b * 3 + t] + 1.0f);
        }
    }
}

} // namespace smart_dnn

#endif // TEST_CHARACTER_DATASET_CPP

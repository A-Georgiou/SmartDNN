#ifndef TEST_MULTI_HEAD_ATTENTION_CPP
#define TEST_MULTI_HEAD_ATTENTION_CPP

#include <gtest/gtest.h>
#include <cmath>
#include "../../smart_dnn/Debugging/GradientCheck.hpp"
#include "../../smart_dnn/Layers/MultiHeadAttentionLayer.hpp"

namespace smart_dnn {

namespace {

Tensor<float> wave(const Shape& shape, float frequency, float phase) {
    Tensor<float> tensor(shape, 0.0f);
    for (size_t i = 0; i < shape.size(); ++i) {
        tensor[i] = std::sin(frequency * static_cast<float>(i) + phase);
    }
    return tensor;
}

} // namespace

TEST(MultiHeadAttentionLayerTest, PassesGradientCheckWithAndWithoutTheCausalMask) {
    for (bool causal : {true, false}) {
        MultiHeadAttentionLayer<float> attention(6, 2, causal);

        for (const GradientCheckResult<float>& result : checkLayerGradients(attention, wave(Shape({2, 4, 6}), 0.7f, 0.0f), wave(Shape({2, 4, 6}), 1.3f, 0.5f))) {
            EXPECT_TRUE(result.passed) << (causal ? "causal" : "full") << " index " << result.index
                << ": analytic " << result.analytic << ", numeric " << result.numeric;
        }
    }
}

TEST(MultiHeadAttentionLayerTest, CausalOutputIgnoresLaterPositions) {
    MultiHeadAttentionLayer<float> attention(4, 2);
    Tensor<float> input = wave(Shape({1, 3, 4}), 0.7f, 0.0f);
    const Tensor<float> before = attention.forward(input);
    for (size_t j = 8; j < 12; ++j) {
        input[j] += 5.0f;
    }

    const Tensor<float> after = attention.forward(input);

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(after[i], before[i]) << "earlier position changed at " << i;
    }
    EXPECT_NE(after[8], before[8]);
}

TEST(MultiHeadAttentionLayerTest, RejectsInvalidShapes) {
    EXPECT_THROW(MultiHeadAttentionLayer<float>(6, 4), std::invalid_argument);
    MultiHeadAttentionLayer<float> attention(4, 2);
    EXPECT_THROW(attention.forward(Tensor<float>({3, 4}, 0.0f)), std::invalid_argument);
}

} // namespace smart_dnn

#endif // TEST_MULTI_HEAD_ATTENTION_CPP

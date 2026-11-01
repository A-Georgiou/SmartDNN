#ifndef TEST_SOFTMAX_CROSS_ENTROPY_LOSS_CPP
#define TEST_SOFTMAX_CROSS_ENTROPY_LOSS_CPP

#include <gtest/gtest.h>
#include <cmath>
#include "../utils/tensor_helpers.hpp"
#include "../../smart_dnn/Loss/SoftmaxCrossEntropyLoss.hpp"

namespace smart_dnn {

TEST(SoftmaxCrossEntropyLossTest, MatchesLogSoftmaxAndItsGradient) {
    // Row 0 has probabilities (1/5, 3/5, 1/5) and target 1; row 1 has (2/5, 1/5, 2/5) and target 0.
    Tensor<float> logits({2, 3}, {0.0f, std::log(3.0f), 0.0f, std::log(2.0f), 0.0f, std::log(2.0f)});
    Tensor<float> targets({2}, {1.0f, 0.0f});
    SoftmaxCrossEntropyLoss<float> loss;

    EXPECT_NEAR(loss.compute(logits, targets)[0], (std::log(5.0f / 3.0f) + std::log(5.0f / 2.0f)) / 2.0f, 1e-6);
    EXPECT_TRUE(TensorEquals(loss.gradient(logits, targets), Tensor<float>({2, 3}, {0.1f, -0.2f, 0.1f, -0.3f, 0.1f, 0.2f})));
    // Large equal logits still give log(2) rather than losing it next to the max.
    EXPECT_NEAR(loss.compute(Tensor<float>({1, 2}, {1e8f, 1e8f}), Tensor<float>({1}, {0.0f}))[0], std::log(2.0f), 1e-6);
}

TEST(SoftmaxCrossEntropyLossTest, SkipsIgnoredPositionsInSequences) {
    // (batch 1, time 2, classes 3) where the second position is padding.
    Tensor<float> logits({1, 2, 3}, {0.0f, std::log(3.0f), 0.0f, 5.0f, -5.0f, 1.0f});
    Tensor<float> targets({1, 2}, {1.0f, -1.0f});
    SoftmaxCrossEntropyLoss<float> loss;

    EXPECT_NEAR(loss.compute(logits, targets)[0], std::log(5.0f / 3.0f), 1e-6);
    EXPECT_TRUE(TensorEquals(loss.gradient(logits, targets), Tensor<float>({1, 2, 3}, {0.2f, -0.4f, 0.2f, 0.0f, 0.0f, 0.0f})));
}

TEST(SoftmaxCrossEntropyLossTest, RejectsInvalidTargets) {
    Tensor<float> logits({2, 3}, 0.0f);
    SoftmaxCrossEntropyLoss<float> loss;

    EXPECT_THROW(loss.compute(logits, Tensor<float>({2}, {0.0f, 3.0f})), std::invalid_argument);
    EXPECT_THROW(loss.compute(logits, Tensor<float>({2}, {0.0f, 1.5f})), std::invalid_argument);
    EXPECT_THROW(loss.gradient(logits, Tensor<float>({2, 3}, 0.0f)), std::invalid_argument);
}

} // namespace smart_dnn

#endif // TEST_SOFTMAX_CROSS_ENTROPY_LOSS_CPP

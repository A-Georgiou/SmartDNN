#ifndef TEST_LOSS_FUNCTIONS_CPP
#define TEST_LOSS_FUNCTIONS_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Loss/MSELoss.hpp"
#include "../../smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include <cmath>

namespace smart_dnn {

// Helper function to check if two floats are approximately equal
static inline bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// ========== MSE Loss Tests ==========

TEST(MSELossTest, ComputeLossBasic) {
    MSELoss<float> mseLoss;
    
    // Create simple prediction and target
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 2}, {1.5f, 2.5f, 2.5f, 3.5f});
    
    // Compute loss: ((0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2) / 4 = 0.25
    Tensor<float> loss = mseLoss.compute(prediction, target);
    
    ASSERT_EQ(loss.getShape(), Shape({1}));
    EXPECT_TRUE(approxEqual(loss.getData()[0], 0.25f));
}

TEST(MSELossTest, ComputeLossZeroError) {
    MSELoss<float> mseLoss;
    
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    Tensor<float> loss = mseLoss.compute(prediction, target);
    
    ASSERT_EQ(loss.getShape(), Shape({1}));
    EXPECT_TRUE(approxEqual(loss.getData()[0], 0.0f));
}

TEST(MSELossTest, ComputeLossLargeError) {
    MSELoss<float> mseLoss;
    
    Tensor<float> prediction({1, 3}, {0.0f, 0.0f, 0.0f});
    Tensor<float> target({1, 3}, {1.0f, 2.0f, 3.0f});
    
    // Loss: ((1)^2 + (2)^2 + (3)^2) / 3 = 14/3 ≈ 4.6667
    Tensor<float> loss = mseLoss.compute(prediction, target);
    
    EXPECT_TRUE(approxEqual(loss.getData()[0], 14.0f/3.0f, 1e-4f));
}

TEST(MSELossTest, GradientBasic) {
    MSELoss<float> mseLoss;
    
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 2}, {1.5f, 2.5f, 2.5f, 3.5f});
    
    // Gradient: 2 * (prediction - target) / n
    // = 2 * [-0.5, -0.5, 0.5, 0.5] / 4 = [-0.25, -0.25, 0.25, 0.25]
    Tensor<float> grad = mseLoss.gradient(prediction, target);
    
    ASSERT_EQ(grad.getShape(), prediction.getShape());
    EXPECT_TRUE(approxEqual(grad.getData()[0], -0.25f));
    EXPECT_TRUE(approxEqual(grad.getData()[1], -0.25f));
    EXPECT_TRUE(approxEqual(grad.getData()[2], 0.25f));
    EXPECT_TRUE(approxEqual(grad.getData()[3], 0.25f));
}

TEST(MSELossTest, GradientZeroError) {
    MSELoss<float> mseLoss;
    
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    Tensor<float> grad = mseLoss.gradient(prediction, target);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(approxEqual(grad.getData()[i], 0.0f));
    }
}

TEST(MSELossTest, ShapeMismatchThrows) {
    MSELoss<float> mseLoss;
    
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    EXPECT_THROW(mseLoss.compute(prediction, target), std::invalid_argument);
    EXPECT_THROW(mseLoss.gradient(prediction, target), std::invalid_argument);
}

TEST(MSELossTest, ReshapeTarget1DTo2D) {
    MSELoss<float> mseLoss;
    
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({4}, {1.5f, 2.5f, 3.5f, 4.5f});
    
    // Should reshape target from 1D to 2D if batch sizes match
    EXPECT_NO_THROW({
        Tensor<float> loss = mseLoss.compute(prediction, target);
        Tensor<float> grad = mseLoss.gradient(prediction, target);
    });
}

// ========== Categorical Cross Entropy Loss Tests ==========

TEST(CategoricalCrossEntropyLossTest, ComputeLossBasic) {
    CategoricalCrossEntropyLoss<float> cceLoss;
    
    // Perfect prediction scenario: prediction matches target
    Tensor<float> prediction({2, 3}, {1.0f, 0.0f, 0.0f,  // Class 0
                                       0.0f, 1.0f, 0.0f}); // Class 1
    Tensor<float> target({2, 3}, {1.0f, 0.0f, 0.0f,
                                   0.0f, 1.0f, 0.0f});
    
    Tensor<float> loss = cceLoss.compute(prediction, target);
    
    ASSERT_EQ(loss.getShape(), Shape({1}));
    // Loss should be close to 0 for perfect prediction (with epsilon for numerical stability)
    EXPECT_TRUE(loss.getData()[0] < 0.01f);
}

TEST(CategoricalCrossEntropyLossTest, ComputeLossImperfect) {
    CategoricalCrossEntropyLoss<float> cceLoss;
    
    // Softmax-like predictions
    Tensor<float> prediction({1, 3}, {0.7f, 0.2f, 0.1f});
    Tensor<float> target({1, 3}, {1.0f, 0.0f, 0.0f});
    
    // Loss: -log(0.7) ≈ 0.357
    Tensor<float> loss = cceLoss.compute(prediction, target);
    
    float expectedLoss = -std::log(0.7f);
    EXPECT_TRUE(approxEqual(loss.getData()[0], expectedLoss, 1e-3f));
}

TEST(CategoricalCrossEntropyLossTest, ComputeLossBatch) {
    CategoricalCrossEntropyLoss<float> cceLoss;
    
    // Batch of 2 samples with 3 classes
    Tensor<float> prediction({2, 3}, {0.8f, 0.1f, 0.1f,
                                       0.2f, 0.7f, 0.1f});
    Tensor<float> target({2, 3}, {1.0f, 0.0f, 0.0f,
                                   0.0f, 1.0f, 0.0f});
    
    // Loss: (-log(0.8) + -log(0.7)) / 2
    Tensor<float> loss = cceLoss.compute(prediction, target);
    
    float expectedLoss = (-std::log(0.8f) - std::log(0.7f)) / 2.0f;
    EXPECT_TRUE(approxEqual(loss.getData()[0], expectedLoss, 1e-3f));
}

TEST(CategoricalCrossEntropyLossTest, GradientBasic) {
    CategoricalCrossEntropyLoss<float> cceLoss;
    
    Tensor<float> prediction({2, 2}, {0.8f, 0.2f,
                                       0.3f, 0.7f});
    Tensor<float> target({2, 2}, {1.0f, 0.0f,
                                   0.0f, 1.0f});
    
    Tensor<float> grad = cceLoss.gradient(prediction, target);
    
    ASSERT_EQ(grad.getShape(), prediction.getShape());
    
    // Gradient: (prediction - target) / batchSize
    float batchSize = 2.0f;
    EXPECT_TRUE(approxEqual(grad.getData()[0], (0.8f - 1.0f) / batchSize));
    EXPECT_TRUE(approxEqual(grad.getData()[1], (0.2f - 0.0f) / batchSize));
    EXPECT_TRUE(approxEqual(grad.getData()[2], (0.3f - 0.0f) / batchSize));
    EXPECT_TRUE(approxEqual(grad.getData()[3], (0.7f - 1.0f) / batchSize));
}

TEST(CategoricalCrossEntropyLossTest, GradientPerfectPrediction) {
    CategoricalCrossEntropyLoss<float> cceLoss;
    
    Tensor<float> prediction({2, 2}, {1.0f, 0.0f,
                                       0.0f, 1.0f});
    Tensor<float> target({2, 2}, {1.0f, 0.0f,
                                   0.0f, 1.0f});
    
    Tensor<float> grad = cceLoss.gradient(prediction, target);
    
    // All gradients should be close to 0 for perfect prediction
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(approxEqual(grad.getData()[i], 0.0f, 1e-2f));
    }
}

TEST(CategoricalCrossEntropyLossTest, ShapeMismatchThrows) {
    CategoricalCrossEntropyLoss<float> cceLoss;
    
    Tensor<float> prediction({2, 3}, {0.8f, 0.1f, 0.1f,
                                       0.2f, 0.7f, 0.1f});
    Tensor<float> target({2, 2}, {1.0f, 0.0f,
                                   0.0f, 1.0f});
    
    EXPECT_THROW(cceLoss.compute(prediction, target), std::invalid_argument);
    EXPECT_THROW(cceLoss.gradient(prediction, target), std::invalid_argument);
}

TEST(CategoricalCrossEntropyLossTest, NumericalStability) {
    CategoricalCrossEntropyLoss<float> cceLoss;
    
    // Test with extreme values (close to 0 and 1)
    Tensor<float> prediction({1, 2}, {0.9999f, 0.0001f});
    Tensor<float> target({1, 2}, {1.0f, 0.0f});
    
    // Should not throw or produce NaN/Inf due to epsilon handling
    EXPECT_NO_THROW({
        Tensor<float> loss = cceLoss.compute(prediction, target);
        EXPECT_FALSE(std::isnan(loss.getData()[0]));
        EXPECT_FALSE(std::isinf(loss.getData()[0]));
        
        Tensor<float> grad = cceLoss.gradient(prediction, target);
        for (size_t i = 0; i < grad.getShape().size(); ++i) {
            EXPECT_FALSE(std::isnan(grad.getData()[i]));
            EXPECT_FALSE(std::isinf(grad.getData()[i]));
        }
    });
}

} // namespace smart_dnn

#endif // TEST_LOSS_FUNCTIONS_CPP

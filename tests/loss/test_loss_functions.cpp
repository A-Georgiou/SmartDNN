#ifndef TEST_LOSS_FUNCTIONS_CPP
#define TEST_LOSS_FUNCTIONS_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Loss/MSELoss.hpp"
#include "../../smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include "../utils/tensor_helpers.hpp"
#include <cmath>

namespace smart_dnn {

// ============================================================================
// MSELoss Tests
// ============================================================================

TEST(MSELossTest, ComputeWithMatchingShapes) {
    MSELoss<float> loss;
    
    // Create simple predictions and targets
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 2}, {1.5f, 2.5f, 3.5f, 4.5f});
    
    // MSE = mean((pred - target)^2)
    // diff = [-0.5, -0.5, -0.5, -0.5]
    // squared = [0.25, 0.25, 0.25, 0.25]
    // mean = 1.0 / 4 = 0.25
    Tensor<float> result = loss.compute(prediction, target);
    
    EXPECT_EQ(result.getShape().size(), 1);
    EXPECT_TRUE(approxEqual(result.getData()[0], 0.25f));
}

TEST(MSELossTest, ComputeWithPerfectPrediction) {
    MSELoss<float> loss;
    
    Tensor<float> prediction({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> target({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    Tensor<float> result = loss.compute(prediction, target);
    
    EXPECT_TRUE(approxEqual(result.getData()[0], 0.0f));
}

TEST(MSELossTest, ComputeWithReshapableTarget) {
    MSELoss<float> loss;
    
    // Target is 1D but can be reshaped to match prediction
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2}, {1.0f, 2.0f});  // Cannot be reshaped to (2, 2)
    
    // Should throw because size mismatch (2 != 4)
    EXPECT_THROW(loss.compute(prediction, target), std::runtime_error);
}

TEST(MSELossTest, ComputeWithMismatchedShapes) {
    MSELoss<float> loss;
    
    Tensor<float> prediction({2, 3}, 1.0f);
    Tensor<float> target({3, 2}, 1.0f);
    
    EXPECT_THROW(loss.compute(prediction, target), std::invalid_argument);
}

TEST(MSELossTest, GradientComputation) {
    MSELoss<float> loss;
    
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 2}, {1.5f, 2.5f, 3.5f, 4.5f});
    
    // Gradient = 2 * (pred - target) / size
    // diff = [-0.5, -0.5, -0.5, -0.5]
    // grad = 2 * diff / 4 = [-0.25, -0.25, -0.25, -0.25]
    Tensor<float> grad = loss.gradient(prediction, target);
    
    EXPECT_EQ(grad.getShape(), prediction.getShape());
    
    const float* gradData = grad.getData().data();
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(approxEqual(gradData[i], -0.25f));
    }
}

TEST(MSELossTest, GradientWithZeroDifference) {
    MSELoss<float> loss;
    
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    Tensor<float> grad = loss.gradient(prediction, target);
    
    const float* gradData = grad.getData().data();
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(approxEqual(gradData[i], 0.0f));
    }
}

// ============================================================================
// CategoricalCrossEntropyLoss Tests
// ============================================================================

TEST(CategoricalCrossEntropyLossTest, ComputeBasic) {
    CategoricalCrossEntropyLoss<float> loss;
    
    // Batch size 2, 3 classes
    // First sample: true class is 1 (one-hot: [0, 1, 0])
    // Second sample: true class is 2 (one-hot: [0, 0, 1])
    Tensor<float> prediction({2, 3}, {0.1f, 0.7f, 0.2f,   // First sample predictions
                                       0.2f, 0.3f, 0.5f}); // Second sample predictions
    Tensor<float> target({2, 3}, {0.0f, 1.0f, 0.0f,       // First sample target
                                   0.0f, 0.0f, 1.0f});     // Second sample target
    
    // Loss = -mean(sum(target * log(prediction)))
    // Sample 1: -1.0 * log(0.7) ≈ 0.357
    // Sample 2: -1.0 * log(0.5) ≈ 0.693
    // Average: (0.357 + 0.693) / 2 ≈ 0.525
    Tensor<float> result = loss.compute(prediction, target);
    
    EXPECT_EQ(result.getShape().size(), 1);
    
    // The actual expected value
    float expected = (std::log(0.7f) + std::log(0.5f)) / -2.0f;
    EXPECT_TRUE(approxEqual(result.getData()[0], expected, 1e-4f));
}

TEST(CategoricalCrossEntropyLossTest, ComputeWithPerfectPrediction) {
    CategoricalCrossEntropyLoss<float> loss;
    
    // Perfect predictions (except for epsilon to prevent log(0))
    Tensor<float> prediction({2, 3}, {0.0f, 1.0f, 0.0f,
                                       0.0f, 0.0f, 1.0f});
    Tensor<float> target({2, 3}, {0.0f, 1.0f, 0.0f,
                                   0.0f, 0.0f, 1.0f});
    
    Tensor<float> result = loss.compute(prediction, target);
    
    // Should be very close to 0 (with epsilon handling)
    EXPECT_LT(result.getData()[0], 1e-5f);
}

TEST(CategoricalCrossEntropyLossTest, ComputeWithMismatchedShapes) {
    CategoricalCrossEntropyLoss<float> loss;
    
    Tensor<float> prediction({2, 3}, 0.5f);
    Tensor<float> target({3, 2}, 0.5f);
    
    EXPECT_THROW(loss.compute(prediction, target), std::invalid_argument);
}

TEST(CategoricalCrossEntropyLossTest, GradientComputation) {
    CategoricalCrossEntropyLoss<float> loss;
    
    Tensor<float> prediction({2, 3}, {0.1f, 0.7f, 0.2f,
                                       0.2f, 0.3f, 0.5f});
    Tensor<float> target({2, 3}, {0.0f, 1.0f, 0.0f,
                                   0.0f, 0.0f, 1.0f});
    
    Tensor<float> grad = loss.gradient(prediction, target);
    
    EXPECT_EQ(grad.getShape(), prediction.getShape());
    
    // Gradient = (prediction - target) / batchSize
    // For first sample, class 0: (0.1 - 0.0) / 2 = 0.05
    // For first sample, class 1: (0.7 - 1.0) / 2 = -0.15
    // For first sample, class 2: (0.2 - 0.0) / 2 = 0.1
    const float* gradData = grad.getData().data();
    EXPECT_TRUE(approxEqual(gradData[0], 0.05f, 1e-4f));
    EXPECT_TRUE(approxEqual(gradData[1], -0.15f, 1e-4f));
    EXPECT_TRUE(approxEqual(gradData[2], 0.1f, 1e-4f));
}

TEST(CategoricalCrossEntropyLossTest, GradientWithMismatchedShapes) {
    CategoricalCrossEntropyLoss<float> loss;
    
    Tensor<float> prediction({2, 3}, 0.5f);
    Tensor<float> target({3, 2}, 0.5f);
    
    EXPECT_THROW(loss.gradient(prediction, target), std::invalid_argument);
}

TEST(CategoricalCrossEntropyLossTest, NumericalStability) {
    CategoricalCrossEntropyLoss<float> loss;
    
    // Test with extreme values that could cause log(0) or log(1)
    Tensor<float> prediction({1, 2}, {0.0f, 1.0f});  // Extreme values
    Tensor<float> target({1, 2}, {0.0f, 1.0f});
    
    // Should not throw and should handle epsilon properly
    EXPECT_NO_THROW({
        Tensor<float> result = loss.compute(prediction, target);
        Tensor<float> grad = loss.gradient(prediction, target);
    });
}

} // namespace smart_dnn

#endif // TEST_LOSS_FUNCTIONS_CPP

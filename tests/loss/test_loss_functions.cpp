#ifndef TEST_LOSS_FUNCTIONS_CPP
#define TEST_LOSS_FUNCTIONS_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Loss/MSELoss.hpp"
#include "../../smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include "../utils/tensor_helpers.hpp"
#include <cmath>

namespace smart_dnn {

// Helper function to check if two floats are approximately equal
static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// ==================== MSELoss Tests ====================

TEST(MSELossTest, ComputeLossWithMatchingShapes) {
    MSELoss<float> mse;
    
    // Create prediction and target tensors
    Tensor<float> prediction({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> target({2, 3}, {1.5f, 2.5f, 2.5f, 3.5f, 4.5f, 5.5f});
    
    // Compute loss
    Tensor<float> loss = mse.compute(prediction, target);
    
    // Expected: mean of squared differences
    // Differences: [-0.5, -0.5, 0.5, 0.5, 0.5, 0.5]
    // Squared: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    // Mean: 0.25
    ASSERT_EQ(loss.getShape().size(), 1);
    EXPECT_TRUE(approxEqual(loss.getData()[0], 0.25f));
}

TEST(MSELossTest, ComputeLossWithZeroDifference) {
    MSELoss<float> mse;
    
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    Tensor<float> loss = mse.compute(prediction, target);
    
    // Loss should be zero when prediction equals target
    EXPECT_TRUE(approxEqual(loss.getData()[0], 0.0f));
}

TEST(MSELossTest, ComputeLossWithIncompatibleShapes) {
    MSELoss<float> mse;
    
    Tensor<float> prediction({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> target({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    // Should throw for incompatible shapes
    EXPECT_THROW(mse.compute(prediction, target), std::invalid_argument);
}

TEST(MSELossTest, GradientWithMatchingShapes) {
    MSELoss<float> mse;
    
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 2}, {0.5f, 1.5f, 2.5f, 3.5f});
    
    Tensor<float> grad = mse.gradient(prediction, target);
    
    // Gradient: 2 * (prediction - target) / size
    // Differences: [0.5, 0.5, 0.5, 0.5]
    // Gradient: 2 * [0.5, 0.5, 0.5, 0.5] / 4 = [0.25, 0.25, 0.25, 0.25]
    ASSERT_EQ(grad.getShape(), prediction.getShape());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(approxEqual(grad.getData()[i], 0.25f));
    }
}

TEST(MSELossTest, GradientWithZeroDifference) {
    MSELoss<float> mse;
    
    Tensor<float> prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    Tensor<float> grad = mse.gradient(prediction, target);
    
    // Gradient should be zero when prediction equals target
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(approxEqual(grad.getData()[i], 0.0f));
    }
}

// ==================== CategoricalCrossEntropyLoss Tests ====================

TEST(CategoricalCrossEntropyLossTest, ComputeLossBasic) {
    CategoricalCrossEntropyLoss<float> cce;
    
    // Create prediction (2 samples, 3 classes)
    // Using probabilities that sum to 1 for each sample
    Tensor<float> prediction({2, 3}, {
        0.7f, 0.2f, 0.1f,  // First sample
        0.1f, 0.8f, 0.1f   // Second sample
    });
    
    // One-hot encoded targets
    Tensor<float> target({2, 3}, {
        1.0f, 0.0f, 0.0f,  // First sample: class 0
        0.0f, 1.0f, 0.0f   // Second sample: class 1
    });
    
    Tensor<float> loss = cce.compute(prediction, target);
    
    // Expected: -mean(target * log(prediction))
    // Sample 1: -1.0 * log(0.7) ≈ 0.3567
    // Sample 2: -1.0 * log(0.8) ≈ 0.2231
    // Mean: (0.3567 + 0.2231) / 2 ≈ 0.2899
    ASSERT_EQ(loss.getShape(), Shape({1}));
    EXPECT_TRUE(approxEqual(loss.getData()[0], 0.2899f, 1e-3f));
}

TEST(CategoricalCrossEntropyLossTest, ComputeLossWithPerfectPrediction) {
    CategoricalCrossEntropyLoss<float> cce;
    
    Tensor<float> prediction({2, 3}, {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    });
    
    Tensor<float> target({2, 3}, {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    });
    
    Tensor<float> loss = cce.compute(prediction, target);
    
    // Loss should be very close to zero for perfect prediction
    // Due to epsilon clamping, it won't be exactly zero
    EXPECT_LT(loss.getData()[0], 0.0001f);
}

TEST(CategoricalCrossEntropyLossTest, ComputeLossWithIncompatibleShapes) {
    CategoricalCrossEntropyLoss<float> cce;
    
    Tensor<float> prediction({2, 3}, {0.5f, 0.3f, 0.2f, 0.4f, 0.4f, 0.2f});
    Tensor<float> target({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    
    // Should throw for mismatched shapes
    EXPECT_THROW(cce.compute(prediction, target), std::invalid_argument);
}

TEST(CategoricalCrossEntropyLossTest, GradientBasic) {
    CategoricalCrossEntropyLoss<float> cce;
    
    Tensor<float> prediction({2, 3}, {
        0.7f, 0.2f, 0.1f,
        0.1f, 0.8f, 0.1f
    });
    
    Tensor<float> target({2, 3}, {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    });
    
    Tensor<float> grad = cce.gradient(prediction, target);
    
    ASSERT_EQ(grad.getShape(), prediction.getShape());
    
    // Gradient: (prediction - target) / batchSize
    // Sample 1: [(0.7-1.0), (0.2-0.0), (0.1-0.0)] = [-0.3, 0.2, 0.1]
    // Sample 2: [(0.1-0.0), (0.8-1.0), (0.1-0.0)] = [0.1, -0.2, 0.1]
    // Divided by batch size (2)
    std::vector<float> expected = {-0.15f, 0.1f, 0.05f, 0.05f, -0.1f, 0.05f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_TRUE(approxEqual(grad.getData()[i], expected[i], 1e-4f));
    }
}

TEST(CategoricalCrossEntropyLossTest, GradientWithPerfectPrediction) {
    CategoricalCrossEntropyLoss<float> cce;
    
    Tensor<float> prediction({1, 3}, {1.0f, 0.0f, 0.0f});
    Tensor<float> target({1, 3}, {1.0f, 0.0f, 0.0f});
    
    Tensor<float> grad = cce.gradient(prediction, target);
    
    // Gradients should be very close to zero for perfect prediction
    // Due to epsilon clamping in the implementation
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_LT(std::abs(grad.getData()[i]), 0.0001f);
    }
}

TEST(CategoricalCrossEntropyLossTest, GradientWithIncompatibleShapes) {
    CategoricalCrossEntropyLoss<float> cce;
    
    Tensor<float> prediction({2, 3}, {0.5f, 0.3f, 0.2f, 0.4f, 0.4f, 0.2f});
    Tensor<float> target({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    
    // Should throw for mismatched shapes
    EXPECT_THROW(cce.gradient(prediction, target), std::invalid_argument);
}

} // namespace smart_dnn

#endif // TEST_LOSS_FUNCTIONS_CPP

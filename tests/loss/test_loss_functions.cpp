#ifndef TEST_LOSS_FUNCTIONS_CPP
#define TEST_LOSS_FUNCTIONS_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Loss/MSELoss.hpp"
#include "../../smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include "../utils/tensor_helpers.hpp"
#include <cmath>

namespace smart_dnn {

// ==================== MSE Loss Tests ====================

TEST(MSELossTest, ComputeLossSimple) {
    MSELoss<float> loss;
    
    // Simple case: prediction = [1, 2, 3], target = [1, 1, 1]
    Tensor<float> prediction({3}, {1.0f, 2.0f, 3.0f});
    Tensor<float> target({3}, {1.0f, 1.0f, 1.0f});
    
    Tensor<float> result = loss.compute(prediction, target);
    
    // MSE = ((1-1)^2 + (2-1)^2 + (3-1)^2) / 3 = (0 + 1 + 4) / 3 = 5/3 ≈ 1.6667
    EXPECT_NEAR(result.getData()[0], 5.0f/3.0f, 1e-5f);
}

TEST(MSELossTest, ComputeLossPerfectPrediction) {
    MSELoss<float> loss;
    
    Tensor<float> prediction({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> target({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    Tensor<float> result = loss.compute(prediction, target);
    
    // MSE should be 0 for perfect prediction
    EXPECT_NEAR(result.getData()[0], 0.0f, 1e-6f);
}

TEST(MSELossTest, ComputeLossBatched) {
    MSELoss<float> loss;
    
    // Batch of 2 samples, each with 3 features
    Tensor<float> prediction({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> target({2, 3}, {1.0f, 1.0f, 1.0f, 4.0f, 4.0f, 4.0f});
    
    Tensor<float> result = loss.compute(prediction, target);
    
    // MSE = ((0)^2 + (1)^2 + (2)^2 + (0)^2 + (1)^2 + (2)^2) / 6 = 10/6 ≈ 1.6667
    EXPECT_NEAR(result.getData()[0], 10.0f/6.0f, 1e-5f);
}

TEST(MSELossTest, GradientSimple) {
    MSELoss<float> loss;
    
    Tensor<float> prediction({3}, {2.0f, 3.0f, 4.0f});
    Tensor<float> target({3}, {1.0f, 1.0f, 1.0f});
    
    Tensor<float> grad = loss.gradient(prediction, target);
    
    // Gradient = 2 * (prediction - target) / n
    // For each element: 2 * (pred[i] - target[i]) / 3
    EXPECT_NEAR(grad.getData()[0], 2.0f * (2.0f - 1.0f) / 3.0f, 1e-5f);  // 2/3
    EXPECT_NEAR(grad.getData()[1], 2.0f * (3.0f - 1.0f) / 3.0f, 1e-5f);  // 4/3
    EXPECT_NEAR(grad.getData()[2], 2.0f * (4.0f - 1.0f) / 3.0f, 1e-5f);  // 2
}

TEST(MSELossTest, GradientPerfectPrediction) {
    MSELoss<float> loss;
    
    Tensor<float> prediction({3}, {1.0f, 2.0f, 3.0f});
    Tensor<float> target({3}, {1.0f, 2.0f, 3.0f});
    
    Tensor<float> grad = loss.gradient(prediction, target);
    
    // Gradient should be 0 for perfect prediction
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(grad.getData()[i], 0.0f, 1e-6f);
    }
}

TEST(MSELossTest, ShapeMismatchThrows) {
    MSELoss<float> loss;
    
    Tensor<float> prediction({3}, {1.0f, 2.0f, 3.0f});
    Tensor<float> target({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    EXPECT_THROW(loss.compute(prediction, target), std::invalid_argument);
    EXPECT_THROW(loss.gradient(prediction, target), std::invalid_argument);
}

// ==================== Categorical Cross Entropy Loss Tests ====================

TEST(CategoricalCrossEntropyLossTest, ComputeLossSimple) {
    CategoricalCrossEntropyLoss<float> loss;
    
    // Simple case: 2 samples, 3 classes
    // First sample: class 0 is correct (probability 0.7)
    // Second sample: class 2 is correct (probability 0.6)
    Tensor<float> prediction({2, 3}, {
        0.7f, 0.2f, 0.1f,  // Sample 1 predictions
        0.1f, 0.3f, 0.6f   // Sample 2 predictions
    });
    
    Tensor<float> target({2, 3}, {
        1.0f, 0.0f, 0.0f,  // Sample 1 is class 0
        0.0f, 0.0f, 1.0f   // Sample 2 is class 2
    });
    
    Tensor<float> result = loss.compute(prediction, target);
    
    // Loss = -[1*log(0.7) + 0 + 0 + 0 + 0 + 1*log(0.6)] / 2
    float expected = -(std::log(0.7f) + std::log(0.6f)) / 2.0f;
    EXPECT_NEAR(result.getData()[0], expected, 1e-4f);
}

TEST(CategoricalCrossEntropyLossTest, ComputeLossPerfectPrediction) {
    CategoricalCrossEntropyLoss<float> loss;
    
    // Perfect predictions (probability 1.0 for correct class)
    Tensor<float> prediction({2, 3}, {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    });
    
    Tensor<float> target({2, 3}, {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    });
    
    Tensor<float> result = loss.compute(prediction, target);
    
    // Loss should be very close to 0 (with epsilon clipping)
    EXPECT_NEAR(result.getData()[0], 0.0f, 1e-5f);
}

TEST(CategoricalCrossEntropyLossTest, ComputeLossWorstPrediction) {
    CategoricalCrossEntropyLoss<float> loss;
    
    // Worst case: predicting 0 for correct class (will be clipped to epsilon)
    Tensor<float> prediction({1, 2}, {0.0f, 1.0f});
    Tensor<float> target({1, 2}, {1.0f, 0.0f});
    
    Tensor<float> result = loss.compute(prediction, target);
    
    // Loss = -log(epsilon) where epsilon = 1e-7 (matches CategoricalCrossEntropyLoss implementation)
    constexpr float epsilon = 1e-7f;
    float expected = -std::log(epsilon);
    EXPECT_NEAR(result.getData()[0], expected, 1e-3f);
}

TEST(CategoricalCrossEntropyLossTest, GradientSimple) {
    CategoricalCrossEntropyLoss<float> loss;
    
    Tensor<float> prediction({2, 2}, {
        0.8f, 0.2f,
        0.3f, 0.7f
    });
    
    Tensor<float> target({2, 2}, {
        1.0f, 0.0f,
        0.0f, 1.0f
    });
    
    Tensor<float> grad = loss.gradient(prediction, target);
    
    // Gradient = (prediction - target) / batchSize
    EXPECT_NEAR(grad.getData()[0], (0.8f - 1.0f) / 2.0f, 1e-5f);  // -0.1
    EXPECT_NEAR(grad.getData()[1], (0.2f - 0.0f) / 2.0f, 1e-5f);  // 0.1
    EXPECT_NEAR(grad.getData()[2], (0.3f - 0.0f) / 2.0f, 1e-5f);  // 0.15
    EXPECT_NEAR(grad.getData()[3], (0.7f - 1.0f) / 2.0f, 1e-5f);  // -0.15
}

TEST(CategoricalCrossEntropyLossTest, GradientPerfectPrediction) {
    CategoricalCrossEntropyLoss<float> loss;
    
    Tensor<float> prediction({2, 3}, {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    });
    
    Tensor<float> target({2, 3}, {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    });
    
    Tensor<float> grad = loss.gradient(prediction, target);
    
    // Gradient should be 0 for perfect prediction
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(grad.getData()[i], 0.0f, 1e-6f);
    }
}

TEST(CategoricalCrossEntropyLossTest, ShapeMismatchThrows) {
    CategoricalCrossEntropyLoss<float> loss;
    
    Tensor<float> prediction({2, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    Tensor<float> target({2, 4}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
    
    EXPECT_THROW(loss.compute(prediction, target), std::invalid_argument);
    EXPECT_THROW(loss.gradient(prediction, target), std::invalid_argument);
}

} // namespace smart_dnn

#endif // TEST_LOSS_FUNCTIONS_CPP

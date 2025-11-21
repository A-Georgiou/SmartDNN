#ifndef TEST_ADAM_OPTIMIZER_CPP
#define TEST_ADAM_OPTIMIZER_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "../utils/tensor_helpers.hpp"

namespace smart_dnn {

// Test basic Adam without any regularization
TEST(AdamOptimizerTest, BasicAdamWithoutRegularization) {
    AdamOptions<float> options;
    options.learningRate = 0.001f;
    options.beta1 = 0.9f;
    options.beta2 = 0.999f;
    options.epsilon = 1e-8f;
    
    AdamOptimizer<float> optimizer(options);
    
    // Create a simple weight tensor and gradient
    Tensor<float> weight({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> gradient({2, 2}, {0.1f, 0.2f, 0.3f, 0.4f});
    
    // Store original weights
    std::vector<float> originalWeights = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Apply optimization
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    optimizer.optimize(weights, gradients);
    
    // Verify weights have been updated
    const float* weightData = weight.getData().data();
    for (size_t i = 0; i < originalWeights.size(); ++i) {
        EXPECT_NE(weightData[i], originalWeights[i]);
        // Weights should decrease (gradient descent)
        EXPECT_LT(weightData[i], originalWeights[i]);
    }
}

// Test Adam with multiple iterations
TEST(AdamOptimizerTest, MultipleIterations) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.beta1 = 0.9f;
    options.beta2 = 0.999f;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight({2, 1}, {1.0f, 2.0f});
    Tensor<float> gradient({2, 1}, {0.1f, 0.2f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    // Store initial weights
    float weight0_initial = weight.getData()[0];
    float weight1_initial = weight.getData()[1];
    
    // First optimization step
    optimizer.optimize(weights, gradients);
    float weight0_after1 = weight.getData()[0];
    float weight1_after1 = weight.getData()[1];
    
    // Weights should have decreased
    EXPECT_LT(weight0_after1, weight0_initial);
    EXPECT_LT(weight1_after1, weight1_initial);
    
    // Second optimization step with same gradient
    optimizer.optimize(weights, gradients);
    float weight0_after2 = weight.getData()[0];
    float weight1_after2 = weight.getData()[1];
    
    // Weights should continue to decrease
    EXPECT_LT(weight0_after2, weight0_after1);
    EXPECT_LT(weight1_after2, weight1_after1);
}

// Test Adam with different beta values
TEST(AdamOptimizerTest, DifferentBetaValues) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.beta1 = 0.5f;  // Lower beta1
    options.beta2 = 0.9f;  // Lower beta2
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    float initialWeight = weight.getData()[0];
    optimizer.optimize(weights, gradients);
    
    // Weight should be updated
    EXPECT_NE(weight.getData()[0], initialWeight);
    EXPECT_LT(weight.getData()[0], initialWeight);
}

// Test learning rate override
TEST(AdamOptimizerTest, LearningRateOverride) {
    AdamOptions<float> options;
    options.learningRate = 0.001f;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight1({1, 1}, {1.0f});
    Tensor<float> gradient1({1, 1}, {0.1f});
    Tensor<float> weight2({1, 1}, {1.0f});
    Tensor<float> gradient2({1, 1}, {0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights1 = {std::ref(weight1)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients1 = {std::ref(gradient1)};
    std::vector<std::reference_wrapper<Tensor<float>>> weights2 = {std::ref(weight2)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients2 = {std::ref(gradient2)};
    
    // Optimize with default learning rate
    optimizer.optimize(weights1, gradients1);
    
    // Create a new optimizer instance to reset state
    AdamOptimizer<float> optimizer2(options);
    
    // Optimize with overridden learning rate (10x higher)
    optimizer2.optimize(weights2, gradients2, 0.01f);
    
    // Weight2 should have changed more than weight1 due to higher learning rate
    float change1 = std::abs(1.0f - weight1.getData()[0]);
    float change2 = std::abs(1.0f - weight2.getData()[0]);
    EXPECT_GT(change2, change1);
}

// Test with zero gradient
TEST(AdamOptimizerTest, ZeroGradient) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> gradient({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    std::vector<float> originalWeights = {1.0f, 2.0f, 3.0f, 4.0f};
    optimizer.optimize(weights, gradients);
    
    // With zero gradient, weights should remain approximately the same
    // (small numerical changes may occur due to bias correction)
    const float* weightData = weight.getData().data();
    for (size_t i = 0; i < originalWeights.size(); ++i) {
        EXPECT_NEAR(weightData[i], originalWeights[i], 0.01f);
    }
}

// Test multiple weight tensors
TEST(AdamOptimizerTest, MultipleWeightTensors) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> gradient1({2, 2}, {0.1f, 0.1f, 0.1f, 0.1f});
    Tensor<float> weight2({1, 3}, {5.0f, 6.0f, 7.0f});
    Tensor<float> gradient2({1, 3}, {0.2f, 0.2f, 0.2f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight1), std::ref(weight2)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient1), std::ref(gradient2)};
    
    std::vector<float> originalWeights1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> originalWeights2 = {5.0f, 6.0f, 7.0f};
    
    optimizer.optimize(weights, gradients);
    
    // Both weight tensors should be updated
    const float* weight1Data = weight1.getData().data();
    for (size_t i = 0; i < originalWeights1.size(); ++i) {
        EXPECT_NE(weight1Data[i], originalWeights1[i]);
    }
    
    const float* weight2Data = weight2.getData().data();
    for (size_t i = 0; i < originalWeights2.size(); ++i) {
        EXPECT_NE(weight2Data[i], originalWeights2[i]);
    }
}

// Test size mismatch throws
TEST(AdamOptimizerTest, SizeMismatchThrows) {
    AdamOptions<float> options;
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> gradient1({2, 2}, {0.1f, 0.1f, 0.1f, 0.1f});
    Tensor<float> gradient2({1, 3}, {0.2f, 0.2f, 0.2f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight1)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient1), std::ref(gradient2)};
    
    EXPECT_THROW(optimizer.optimize(weights, gradients), std::invalid_argument);
}

} // namespace smart_dnn

#endif // TEST_ADAM_OPTIMIZER_CPP

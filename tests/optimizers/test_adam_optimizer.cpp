#ifndef TEST_ADAM_OPTIMIZER_CPP
#define TEST_ADAM_OPTIMIZER_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "../utils/tensor_helpers.hpp"

namespace smart_dnn {

// Helper function for approximate equality
static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// Test basic Adam optimization without any special features
TEST(AdamOptimizerTest, BasicAdamUpdate) {
    AdamOptions<float> options;
    options.learningRate = 0.001f;
    options.beta1 = 0.9f;
    options.beta2 = 0.999f;
    options.epsilon = 1e-8f;
    options.batchSize = 1;
    
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
    
    // After first iteration, weights should be updated
    const float* weightData = weight.getData().data();
    for (size_t i = 0; i < originalWeights.size(); ++i) {
        // Weights should have decreased (gradient descent)
        EXPECT_LT(weightData[i], originalWeights[i]);
    }
}

// Test Adam with multiple iterations to verify momentum
TEST(AdamOptimizerTest, MultipleIterations) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.beta1 = 0.9f;
    options.beta2 = 0.999f;
    options.epsilon = 1e-8f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    float prevWeight = weight.getData()[0];
    
    // Run multiple iterations
    for (int i = 0; i < 5; ++i) {
        optimizer.optimize(weights, gradients);
        float currWeight = weight.getData()[0];
        
        // Weight should continue to decrease
        EXPECT_LT(currWeight, prevWeight);
        prevWeight = currWeight;
    }
}

// Test learning rate override
TEST(AdamOptimizerTest, LearningRateOverride) {
    AdamOptions<float> options;
    options.learningRate = 0.001f;
    options.beta1 = 0.9f;
    options.beta2 = 0.999f;
    options.epsilon = 1e-8f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight1({1, 1}, {1.0f});
    Tensor<float> weight2({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.1f});
    
    // First optimization with default learning rate
    std::vector<std::reference_wrapper<Tensor<float>>> weights1 = {std::ref(weight1)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients1 = {std::ref(gradient)};
    optimizer.optimize(weights1, gradients1);
    
    // Create new optimizer for comparison
    AdamOptimizer<float> optimizer2(options);
    
    // Second optimization with override learning rate (10x higher)
    std::vector<std::reference_wrapper<Tensor<float>>> weights2 = {std::ref(weight2)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients2 = {std::ref(gradient)};
    optimizer2.optimize(weights2, gradients2, 0.01f);
    
    // Weight with higher learning rate should have changed more
    float change1 = std::abs(1.0f - weight1.getData()[0]);
    float change2 = std::abs(1.0f - weight2.getData()[0]);
    EXPECT_GT(change2, change1);
}

// Test L1 regularization
TEST(AdamOptimizerTest, L1Regularization) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.beta1 = 0.9f;
    options.beta2 = 0.999f;
    options.epsilon = 1e-8f;
    options.l1Strength = 0.1f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.0f});  // Zero gradient to test pure L1 effect
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    float initialWeight = weight.getData()[0];
    optimizer.optimize(weights, gradients);
    
    // With L1 regularization and positive weight, weight should decrease
    EXPECT_LT(weight.getData()[0], initialWeight);
}

// Test L2 regularization
TEST(AdamOptimizerTest, L2Regularization) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.beta1 = 0.9f;
    options.beta2 = 0.999f;
    options.epsilon = 1e-8f;
    options.l2Strength = 0.1f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.0f});  // Zero gradient to test pure L2 effect
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    float initialWeight = weight.getData()[0];
    optimizer.optimize(weights, gradients);
    
    // With L2 regularization, weight should decrease
    EXPECT_LT(weight.getData()[0], initialWeight);
}

// Test learning rate decay
TEST(AdamOptimizerTest, LearningRateDecay) {
    AdamOptions<float> options;
    options.learningRate = 0.1f;
    options.beta1 = 0.9f;
    options.beta2 = 0.999f;
    options.epsilon = 1e-8f;
    options.decay = 0.01f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    // First iteration
    optimizer.optimize(weights, gradients);
    float change1 = std::abs(1.0f - weight.getData()[0]);
    
    // Reset weight
    weight.at({0, 0}) = 1.0f;
    
    // Run several more iterations
    for (int i = 0; i < 10; ++i) {
        optimizer.optimize(weights, gradients);
    }
    
    // Reset weight and run one more iteration
    weight.at({0, 0}) = 1.0f;
    optimizer.optimize(weights, gradients);
    float change2 = std::abs(1.0f - weight.getData()[0]);
    
    // Later iterations should produce smaller changes due to learning rate decay
    EXPECT_LT(change2, change1);
}

// Test multiple weight tensors
TEST(AdamOptimizerTest, MultipleWeights) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight1({2, 2}, 1.0f);
    Tensor<float> weight2({3, 1}, 2.0f);
    Tensor<float> gradient1({2, 2}, 0.1f);
    Tensor<float> gradient2({3, 1}, 0.2f);
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {
        std::ref(weight1), std::ref(weight2)
    };
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {
        std::ref(gradient1), std::ref(gradient2)
    };
    
    optimizer.optimize(weights, gradients);
    
    // Both weights should be updated
    const float* weight1Data = weight1.getData().data();
    for (int i = 0; i < 4; ++i) {
        EXPECT_LT(weight1Data[i], 1.0f);
    }
    
    const float* weight2Data = weight2.getData().data();
    for (int i = 0; i < 3; ++i) {
        EXPECT_LT(weight2Data[i], 2.0f);
    }
}

// Test size mismatch error
TEST(AdamOptimizerTest, SizeMismatchError) {
    AdamOptions<float> options;
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weight1({1, 1}, 1.0f);
    Tensor<float> weight2({1, 1}, 1.0f);
    Tensor<float> gradient({1, 1}, 0.1f);
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {
        std::ref(weight1), std::ref(weight2)
    };
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {
        std::ref(gradient)
    };
    
    EXPECT_THROW(optimizer.optimize(weights, gradients), std::invalid_argument);
}

} // namespace smart_dnn

#endif // TEST_ADAM_OPTIMIZER_CPP

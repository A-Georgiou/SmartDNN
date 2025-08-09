#ifndef TEST_SGD_OPTIMIZER_CPP
#define TEST_SGD_OPTIMIZER_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Optimizers/SGDOptimizer.hpp"
#include "../utils/tensor_helpers.hpp"

namespace smart_dnn {

// Test basic SGD without momentum
TEST(SGDOptimizerTest, BasicSGDWithoutMomentum) {
    SGDOptions<float> options;
    options.learningRate = 0.1f;
    options.momentum = 0.0f;
    
    SGDOptimizer<float> optimizer(options);
    
    // Create a simple weight tensor and gradient
    Tensor<float> weight({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> gradient({2, 2}, {0.1f, 0.2f, 0.3f, 0.4f});
    
    // Store original weights
    std::vector<float> originalWeights = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Apply optimization
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    optimizer.optimize(weights, gradients);
    
    // Expected weights: w = w - lr * grad
    std::vector<float> expectedWeights = {
        1.0f - 0.1f * 0.1f,  // 0.99
        2.0f - 0.1f * 0.2f,  // 1.98
        3.0f - 0.1f * 0.3f,  // 2.97
        4.0f - 0.1f * 0.4f   // 3.96
    };
    
    const float* weightData = weight.getData().data();
    for (size_t i = 0; i < expectedWeights.size(); ++i) {
        EXPECT_NEAR(weightData[i], expectedWeights[i], 1e-6f);
    }
}

// Test SGD with momentum
TEST(SGDOptimizerTest, SGDWithMomentum) {
    SGDOptions<float> options;
    options.learningRate = 0.1f;
    options.momentum = 0.9f;
    
    SGDOptimizer<float> optimizer(options);
    
    Tensor<float> weight({2, 1}, {1.0f, 2.0f});
    Tensor<float> gradient({2, 1}, {0.1f, 0.2f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    // First optimization step
    optimizer.optimize(weights, gradients);
    
    // Expected after first step: v = 0 * 0.9 + 1.0 * 0.1 = 0.1, w = w - lr * v
    std::vector<float> expectedWeights1 = {
        1.0f - 0.1f * 0.1f,  // 0.99
        2.0f - 0.1f * 0.2f   // 1.98
    };
    
    const float* weightData = weight.getData().data();
    for (size_t i = 0; i < expectedWeights1.size(); ++i) {
        EXPECT_NEAR(weightData[i], expectedWeights1[i], 1e-6f);
    }
    
    // Second optimization step with same gradient
    optimizer.optimize(weights, gradients);
    
    // Expected after second step: v = 0.1 * 0.9 + 1.0 * 0.1 = 0.19, w = w - lr * v
    std::vector<float> expectedWeights2 = {
        expectedWeights1[0] - 0.1f * (0.1f * 0.9f + 1.0f * 0.1f),
        expectedWeights1[1] - 0.1f * (0.2f * 0.9f + 1.0f * 0.2f)
    };
    
    for (size_t i = 0; i < expectedWeights2.size(); ++i) {
        EXPECT_NEAR(weightData[i], expectedWeights2[i], 1e-5f);
    }
}

// Test SGD with Nesterov momentum
TEST(SGDOptimizerTest, SGDWithNesterovMomentum) {
    SGDOptions<float> options;
    options.learningRate = 0.1f;
    options.momentum = 0.9f;
    options.nesterov = true;
    
    SGDOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    // Apply optimization
    optimizer.optimize(weights, gradients);
    
    // For Nesterov: v = 0.9 * 0 + 1.0 * 0.1 = 0.1, grad_nesterov = 0.1 + 0.9 * 0.1 = 0.19
    float expectedWeight = 1.0f - 0.1f * 0.19f; // 0.981
    
    EXPECT_NEAR(weight.getData().data()[0], expectedWeight, 1e-6f);
}

// Test weight decay
TEST(SGDOptimizerTest, SGDWithWeightDecay) {
    SGDOptions<float> options;
    options.learningRate = 0.1f;
    options.momentum = 0.0f;
    options.weightDecay = 0.01f;
    
    SGDOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    optimizer.optimize(weights, gradients);
    
    // grad_with_decay = 0.1 + 0.01 * 1.0 = 0.11
    float expectedWeight = 1.0f - 0.1f * 0.11f; // 0.989
    
    EXPECT_NEAR(weight.getData().data()[0], expectedWeight, 1e-6f);
}

// Test learning rate override
TEST(SGDOptimizerTest, LearningRateOverride) {
    SGDOptions<float> options;
    options.learningRate = 0.1f;
    options.momentum = 0.0f;
    
    SGDOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    // Use override learning rate of 0.2
    optimizer.optimize(weights, gradients, 0.2f);
    
    float expectedWeight = 1.0f - 0.2f * 0.1f; // 0.98
    
    EXPECT_NEAR(weight.getData().data()[0], expectedWeight, 1e-6f);
}

} // namespace smart_dnn

#endif // TEST_SGD_OPTIMIZER_CPP
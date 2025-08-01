#ifndef TEST_RMSPROP_OPTIMIZER_CPP
#define TEST_RMSPROP_OPTIMIZER_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Optimizers/RMSPropOptimizer.hpp"
#include "../utils/tensor_helpers.hpp"

namespace smart_dnn {

// Test basic RMSProp without momentum
TEST(RMSPropOptimizerTest, BasicRMSProp) {
    RMSPropOptions<float> options;
    options.learningRate = 0.1f;
    options.alpha = 0.9f;
    options.epsilon = 1e-8f;
    options.momentum = 0.0f;
    
    RMSPropOptimizer<float> optimizer(options);
    
    Tensor<float> weight({2, 1}, {1.0f, 2.0f});
    Tensor<float> gradient({2, 1}, {0.1f, 0.2f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    // First optimization step
    optimizer.optimize(weights, gradients);
    
    // Expected: sq_avg = 0.9 * 0 + 0.1 * grad^2 = 0.1 * grad^2
    // denom = sqrt(sq_avg + epsilon), update = grad / denom
    // For grad[0] = 0.1: sq_avg = 0.001, denom = sqrt(0.001 + 1e-8) ≈ 0.0316, update ≈ 3.16
    // For grad[1] = 0.2: sq_avg = 0.004, denom = sqrt(0.004 + 1e-8) ≈ 0.0632, update ≈ 3.16
    
    const float* weightData = weight.getData().data();
    
    // Weights should decrease significantly due to RMSProp's adaptive learning rate
    EXPECT_LT(weightData[0], 1.0f);  // Should be less than original
    EXPECT_LT(weightData[1], 2.0f);  // Should be less than original
}

// Test RMSProp with momentum
TEST(RMSPropOptimizerTest, RMSPropWithMomentum) {
    RMSPropOptions<float> options;
    options.learningRate = 0.01f;
    options.alpha = 0.9f;
    options.epsilon = 1e-8f;
    options.momentum = 0.9f;
    
    RMSPropOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    float originalWeight = weight.getData().data()[0];
    
    // First optimization step
    optimizer.optimize(weights, gradients);
    float weightAfterFirst = weight.getData().data()[0];
    
    // Second optimization step with same gradient (momentum should accumulate)
    optimizer.optimize(weights, gradients);
    float weightAfterSecond = weight.getData().data()[0];
    
    // With momentum, the second step should have a larger change
    float firstChange = originalWeight - weightAfterFirst;
    float secondChange = weightAfterFirst - weightAfterSecond;
    
    EXPECT_GT(secondChange, firstChange);  // Momentum should accelerate the change
}

// Test centered RMSProp
TEST(RMSPropOptimizerTest, CenteredRMSProp) {
    RMSPropOptions<float> options;
    options.learningRate = 0.1f;
    options.alpha = 0.9f;
    options.epsilon = 1e-8f;
    options.centered = true;
    
    RMSPropOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    float originalWeight = weight.getData().data()[0];
    
    optimizer.optimize(weights, gradients);
    
    // Should update the weight
    EXPECT_NE(weight.getData().data()[0], originalWeight);
    EXPECT_LT(weight.getData().data()[0], originalWeight);  // Should decrease
}

// Test weight decay
TEST(RMSPropOptimizerTest, RMSPropWithWeightDecay) {
    RMSPropOptions<float> options;
    options.learningRate = 0.01f;
    options.alpha = 0.9f;
    options.epsilon = 1e-8f;
    options.weightDecay = 0.01f;
    
    RMSPropOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.0f});  // Zero gradient, only weight decay should affect
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    float originalWeight = weight.getData().data()[0];
    
    optimizer.optimize(weights, gradients);
    
    // With weight decay and zero gradient, weight should still decrease
    EXPECT_LT(weight.getData().data()[0], originalWeight);
}

// Test learning rate override
TEST(RMSPropOptimizerTest, LearningRateOverride) {
    RMSPropOptions<float> options;
    options.learningRate = 0.01f;  // Low base learning rate
    options.alpha = 0.9f;
    options.epsilon = 1e-8f;
    
    RMSPropOptimizer<float> optimizer(options);
    
    Tensor<float> weight1({1, 1}, {1.0f});
    Tensor<float> weight2({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights1 = {std::ref(weight1)};
    std::vector<std::reference_wrapper<Tensor<float>>> weights2 = {std::ref(weight2)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    // Optimize with default learning rate
    optimizer.optimize(weights1, gradients);
    
    // Reset optimizer state by creating new instance
    RMSPropOptimizer<float> optimizer2(options);
    
    // Optimize with higher learning rate override
    optimizer2.optimize(weights2, gradients, 0.1f);  // 10x higher learning rate
    
    float change1 = 1.0f - weight1.getData().data()[0];
    float change2 = 1.0f - weight2.getData().data()[0];
    
    // Higher learning rate should cause larger change
    EXPECT_GT(change2, change1);
}

// Test numerical stability with very small gradients
TEST(RMSPropOptimizerTest, NumericalStabilitySmallGradients) {
    RMSPropOptions<float> options;
    options.learningRate = 0.1f;
    options.alpha = 0.9f;
    options.epsilon = 1e-8f;
    
    RMSPropOptimizer<float> optimizer(options);
    
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {1e-10f});  // Very small gradient
    
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    
    float originalWeight = weight.getData().data()[0];
    
    optimizer.optimize(weights, gradients);
    
    // Should not cause numerical issues
    EXPECT_TRUE(std::isfinite(weight.getData().data()[0]));
    EXPECT_NE(weight.getData().data()[0], originalWeight);  // Should still update
}

} // namespace smart_dnn

#endif // TEST_RMSPROP_OPTIMIZER_CPP
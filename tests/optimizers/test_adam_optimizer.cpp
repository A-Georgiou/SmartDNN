#ifndef TEST_ADAM_OPTIMIZER_CPP
#define TEST_ADAM_OPTIMIZER_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Optimizers/AdamOptimizer.hpp"
#include <cmath>

namespace smart_dnn {

// Helper function to check if two floats are approximately equal
static inline bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

TEST(AdamOptimizerTest, BasicOptimization) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.beta1 = 0.9f;
    options.beta2 = 0.999f;
    options.epsilon = 1e-8f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    // Create a simple weight tensor
    Tensor<float> weights({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> gradients({2, 2}, {0.1f, 0.2f, 0.3f, 0.4f});
    
    // Store original weights
    const auto& originalWeights = weights.getData();
    std::vector<float> originalWeightsCopy(originalWeights.begin(), originalWeights.end());
    
    // Perform one optimization step
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs = {std::ref(weights)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradientRefs = {std::ref(gradients)};
    
    optimizer.optimize(weightRefs, gradientRefs);
    
    // Weights should have changed
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NE(weights.getData()[i], originalWeightsCopy[i]);
        // With positive gradients, weights should decrease
        EXPECT_LT(weights.getData()[i], originalWeightsCopy[i]);
    }
}

TEST(AdamOptimizerTest, MultipleIterations) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weights({2}, {1.0f, 1.0f});
    Tensor<float> gradients({2}, {0.1f, 0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs = {std::ref(weights)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradientRefs = {std::ref(gradients)};
    
    // Perform multiple optimization steps
    for (int i = 0; i < 10; ++i) {
        optimizer.optimize(weightRefs, gradientRefs);
    }
    
    // After 10 iterations with constant gradient, weights should be significantly reduced
    EXPECT_LT(weights.getData()[0], 0.99f);
    EXPECT_LT(weights.getData()[1], 0.99f);
}

TEST(AdamOptimizerTest, ZeroGradient) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weights({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> gradients({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
    
    const auto& originalWeights = weights.getData();
    std::vector<float> originalWeightsCopy(originalWeights.begin(), originalWeights.end());
    
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs = {std::ref(weights)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradientRefs = {std::ref(gradients)};
    
    optimizer.optimize(weightRefs, gradientRefs);
    
    // With zero gradient, weights should remain mostly unchanged
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(approxEqual(weights.getData()[i], originalWeightsCopy[i], 1e-3f));
    }
}

TEST(AdamOptimizerTest, L2Regularization) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.l2Strength = 0.01f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weights({2}, {1.0f, 2.0f});
    Tensor<float> gradients({2}, {0.0f, 0.0f});
    
    const auto& originalWeights = weights.getData();
    std::vector<float> originalWeightsCopy(originalWeights.begin(), originalWeights.end());
    
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs = {std::ref(weights)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradientRefs = {std::ref(gradients)};
    
    optimizer.optimize(weightRefs, gradientRefs);
    
    // Even with zero gradient, L2 regularization should reduce weights
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_LT(weights.getData()[i], originalWeightsCopy[i]);
    }
}

TEST(AdamOptimizerTest, L1Regularization) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.l1Strength = 0.01f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weights({2}, {1.0f, -1.0f});
    Tensor<float> gradients({2}, {0.0f, 0.0f});
    
    const auto& originalWeights = weights.getData();
    std::vector<float> originalWeightsCopy(originalWeights.begin(), originalWeights.end());
    
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs = {std::ref(weights)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradientRefs = {std::ref(gradients)};
    
    optimizer.optimize(weightRefs, gradientRefs);
    
    // L1 regularization should push weights towards zero
    EXPECT_LT(weights.getData()[0], originalWeightsCopy[0]);  // Positive weight decreases
    EXPECT_GT(weights.getData()[1], originalWeightsCopy[1]);  // Negative weight increases (towards 0)
}

TEST(AdamOptimizerTest, BatchSizeAveraging) {
    AdamOptions<float> options1, options2;
    options1.learningRate = 0.01f;
    options1.batchSize = 1;
    options2.learningRate = 0.01f;
    options2.batchSize = 2;
    
    AdamOptimizer<float> optimizer1(options1);
    AdamOptimizer<float> optimizer2(options2);
    
    // Same initial weights and gradients
    Tensor<float> weights1({2}, {1.0f, 1.0f});
    Tensor<float> weights2({2}, {1.0f, 1.0f});
    Tensor<float> gradients({2}, {0.2f, 0.2f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs1 = {std::ref(weights1)};
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs2 = {std::ref(weights2)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradientRefs = {std::ref(gradients)};
    
    optimizer1.optimize(weightRefs1, gradientRefs);
    optimizer2.optimize(weightRefs2, gradientRefs);
    
    // With batch size 2, the effective gradient is halved, so updates should be smaller
    float delta1 = 1.0f - weights1.getData()[0];
    float delta2 = 1.0f - weights2.getData()[0];
    
    // delta2 should be smaller than delta1 due to batch averaging
    EXPECT_LT(delta2, delta1);
}

TEST(AdamOptimizerTest, LearningRateOverride) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    Tensor<float> weights1({2}, {1.0f, 1.0f});
    Tensor<float> weights2({2}, {1.0f, 1.0f});
    Tensor<float> gradients({2}, {0.1f, 0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs1 = {std::ref(weights1)};
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs2 = {std::ref(weights2)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradientRefs = {std::ref(gradients)};
    
    // Use default learning rate
    optimizer.optimize(weightRefs1, gradientRefs);
    
    // Use overridden learning rate (higher)
    optimizer.optimize(weightRefs2, gradientRefs, 0.1f);
    
    // Higher learning rate should result in larger weight updates
    float delta1 = 1.0f - weights1.getData()[0];
    float delta2 = 1.0f - weights2.getData()[0];
    
    EXPECT_GT(delta2, delta1);
}

TEST(AdamOptimizerTest, MismatchedSizes) {
    AdamOptimizer<float> optimizer;
    
    Tensor<float> weights1({2}, {1.0f, 1.0f});
    Tensor<float> weights2({2}, {2.0f, 2.0f});
    Tensor<float> gradients({3}, {0.1f, 0.1f, 0.1f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs = {std::ref(weights1), std::ref(weights2)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradientRefs = {std::ref(gradients)};
    
    // Should throw due to vector size mismatch (2 weights, 1 gradient)
    EXPECT_THROW(optimizer.optimize(weightRefs, gradientRefs), std::invalid_argument);
}

TEST(AdamOptimizerTest, MultipleWeightTensors) {
    AdamOptimizer<float> optimizer;
    
    Tensor<float> weights1({2}, {1.0f, 1.0f});
    Tensor<float> weights2({2}, {2.0f, 2.0f});
    Tensor<float> gradients1({2}, {0.1f, 0.1f});
    Tensor<float> gradients2({2}, {0.2f, 0.2f});
    
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs = {std::ref(weights1), std::ref(weights2)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradientRefs = {std::ref(gradients1), std::ref(gradients2)};
    
    const auto& originalWeights1 = weights1.getData();
    std::vector<float> originalWeightsCopy1(originalWeights1.begin(), originalWeights1.end());
    const auto& originalWeights2 = weights2.getData();
    std::vector<float> originalWeightsCopy2(originalWeights2.begin(), originalWeights2.end());
    
    optimizer.optimize(weightRefs, gradientRefs);
    
    // Both weight tensors should be updated
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_NE(weights1.getData()[i], originalWeightsCopy1[i]);
        EXPECT_NE(weights2.getData()[i], originalWeightsCopy2[i]);
    }
}

TEST(AdamOptimizerTest, Convergence) {
    AdamOptions<float> options;
    options.learningRate = 0.1f;
    options.batchSize = 1;
    
    AdamOptimizer<float> optimizer(options);
    
    // Simulate optimization towards target value
    Tensor<float> weights({1}, {5.0f});
    float target = 0.0f;
    
    std::vector<std::reference_wrapper<Tensor<float>>> weightRefs = {std::ref(weights)};
    
    float initialValue = weights.getData()[0];
    
    for (int i = 0; i < 100; ++i) {
        // Gradient pointing towards target
        float gradient = weights.getData()[0] - target;
        Tensor<float> gradients({1}, {gradient});
        std::vector<std::reference_wrapper<Tensor<float>>> gradientRefs = {std::ref(gradients)};
        
        optimizer.optimize(weightRefs, gradientRefs);
    }
    
    // After 100 iterations, should be closer to target
    float finalValue = weights.getData()[0];
    EXPECT_LT(std::abs(finalValue - target), std::abs(initialValue - target));
}

} // namespace smart_dnn

#endif // TEST_ADAM_OPTIMIZER_CPP

#ifndef TEST_ADAM_OPTIMIZER_CPP
#define TEST_ADAM_OPTIMIZER_CPP

#include <gtest/gtest.h>
#include <cmath>
#include "../../smart_dnn/Optimizers/AdamOptimizer.hpp"

namespace smart_dnn {

TEST(AdamOptimizerTest, FirstStepMovesEachWeightByTheLearningRate) {
    AdamOptions<float> options;
    options.learningRate = 0.1f;
    AdamOptimizer<float> optimizer(options);
    Tensor<float> weight({3}, {1.0f, -2.0f, 3.0f});
    Tensor<float> gradient({3}, {0.5f, -0.25f, 4.0f});

    optimizer.optimize({std::ref(weight)}, {std::ref(gradient)});

    // Bias-corrected moments make the first step learningRate * sign(gradient).
    EXPECT_NEAR(weight[0], 0.9f, 1e-6f);
    EXPECT_NEAR(weight[1], -1.9f, 1e-6f);
    EXPECT_NEAR(weight[2], 2.9f, 1e-6f);
}

TEST(AdamOptimizerTest, FollowsTheAdamRecurrenceOverSeveralSteps) {
    AdamOptions<float> options;
    options.learningRate = 0.01f;
    AdamOptimizer<float> optimizer(options);
    Tensor<float> weight({2}, {1.0f, -2.0f});
    const float gradients[3][2] = {{0.5f, -0.25f}, {-1.0f, 0.75f}, {0.2f, 0.1f}};
    float expected[2] = {1.0f, -2.0f};
    float m[2] = {0.0f, 0.0f};
    float v[2] = {0.0f, 0.0f};

    for (int t = 1; t <= 3; ++t) {
        Tensor<float> gradient({2}, {gradients[t - 1][0], gradients[t - 1][1]});
        optimizer.optimize({std::ref(weight)}, {std::ref(gradient)});

        for (int i = 0; i < 2; ++i) {
            const float g = gradients[t - 1][i];
            m[i] = 0.9f * m[i] + 0.1f * g;
            v[i] = 0.999f * v[i] + 0.001f * g * g;
            const float mHat = m[i] / (1.0f - std::pow(0.9f, t));
            const float vHat = v[i] / (1.0f - std::pow(0.999f, t));
            expected[i] -= 0.01f * mHat / (std::sqrt(vHat) + 1e-8f);
            EXPECT_NEAR(weight[i], expected[i], 1e-6f) << "step " << t << ", weight " << i;
        }
    }
}

} // namespace smart_dnn

#endif // TEST_ADAM_OPTIMIZER_CPP

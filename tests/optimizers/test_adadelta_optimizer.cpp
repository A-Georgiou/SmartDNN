#ifndef TEST_ADADELTA_OPTIMIZER_CPP
#define TEST_ADADELTA_OPTIMIZER_CPP

#include <cmath>
#include <gtest/gtest.h>
#include "../../smart_dnn/Optimizers/AdaDeltaOptimizer.hpp"

namespace smart_dnn {

TEST(AdaDeltaOptimizerTest, MatchesFirstAndRepeatedUpdatesPerParameter) {
    AdaDeltaOptions<float> options;
    options.learningRate = 1.0f;
    options.decay = 0.5f;
    options.epsilon = 0.5f;
    AdaDeltaOptimizer<float> optimizer(options);

    Tensor<float> weight({1, 2}, {10.0f, 10.0f});
    Tensor<float> gradient({1, 2}, {1.0f, 2.0f});
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};

    float expectedWeights[] = {10.0f, 10.0f};
    float squaredGradients[] = {0.0f, 0.0f};
    float squaredUpdates[] = {0.0f, 0.0f};
    for (int step = 0; step < 2; ++step) {
        for (size_t i = 0; i < 2; ++i) {
            float grad = gradient.getData()[i];
            squaredGradients[i] = options.decay * squaredGradients[i] +
                                  (1.0f - options.decay) * grad * grad;
            float update = std::sqrt(squaredUpdates[i] + options.epsilon) /
                           std::sqrt(squaredGradients[i] + options.epsilon) * grad;
            squaredUpdates[i] = options.decay * squaredUpdates[i] +
                                (1.0f - options.decay) * update * update;
            expectedWeights[i] -= update;
        }
        optimizer.optimize(weights, gradients);
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_NEAR(weight.getData()[i], expectedWeights[i], 1e-6f);
        }
    }
}

TEST(AdaDeltaOptimizerTest, KeepsStateIsolatedBetweenWeightTensors) {
    AdaDeltaOptions<float> options;
    options.decay = 0.5f;
    options.epsilon = 0.5f;
    AdaDeltaOptimizer<float> optimizer(options);

    Tensor<float> firstWeight({1, 1}, {10.0f});
    Tensor<float> secondWeight({1, 1}, {10.0f});
    Tensor<float> firstGradient({1, 1}, {1.0f});
    Tensor<float> secondGradient({1, 1}, {2.0f});
    auto firstWeights = std::vector<std::reference_wrapper<Tensor<float>>>{std::ref(firstWeight)};
    auto secondWeights = std::vector<std::reference_wrapper<Tensor<float>>>{std::ref(secondWeight)};
    auto firstGradients = std::vector<std::reference_wrapper<Tensor<float>>>{std::ref(firstGradient)};
    auto secondGradients = std::vector<std::reference_wrapper<Tensor<float>>>{std::ref(secondGradient)};

    optimizer.optimize(firstWeights, firstGradients);
    optimizer.optimize(firstWeights, firstGradients);
    optimizer.optimize(secondWeights, secondGradients);

    float expectedSecondWeight = 10.0f - std::sqrt(options.epsilon) /
                                std::sqrt((1.0f - options.decay) * 4.0f + options.epsilon) * 2.0f;
    EXPECT_NEAR(secondWeight.getData()[0], expectedSecondWeight, 1e-6f);
    EXPECT_LT(firstWeight.getData()[0], 10.0f);
}

TEST(AdaDeltaOptimizerTest, AppliesPositiveLearningRateOverride) {
    AdaDeltaOptions<float> options;
    options.learningRate = 0.2f;
    options.decay = 0.5f;
    options.epsilon = 0.5f;
    AdaDeltaOptimizer<float> defaultOptimizer(options);
    AdaDeltaOptimizer<float> overrideOptimizer(options);

    Tensor<float> defaultWeight({1, 1}, {10.0f});
    Tensor<float> overrideWeight({1, 1}, {10.0f});
    Tensor<float> gradient({1, 1}, {1.0f});
    auto defaultWeights = std::vector<std::reference_wrapper<Tensor<float>>>{std::ref(defaultWeight)};
    auto overrideWeights = std::vector<std::reference_wrapper<Tensor<float>>>{std::ref(overrideWeight)};
    auto gradients = std::vector<std::reference_wrapper<Tensor<float>>>{std::ref(gradient)};

    defaultOptimizer.optimize(defaultWeights, gradients);
    overrideOptimizer.optimize(overrideWeights, gradients, 0.8f);

    EXPECT_NEAR(10.0f - overrideWeight.getData()[0],
                4.0f * (10.0f - defaultWeight.getData()[0]), 1e-6f);
}

TEST(AdaDeltaOptimizerTest, RejectsMismatchedWeightAndGradientCounts) {
    AdaDeltaOptimizer<float> optimizer;
    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {1.0f});
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};

    EXPECT_THROW(optimizer.optimize(weights, {}), std::invalid_argument);
}

TEST(AdaDeltaOptimizerTest, RemainsFiniteForSmallAndZeroGradients) {
    AdaDeltaOptimizer<float> optimizer;
    Tensor<float> weight({1, 2}, {0.0f, 1.0f});
    Tensor<float> gradient({1, 2}, {1e-20f, 0.0f});
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};

    optimizer.optimize(weights, gradients);

    EXPECT_TRUE(std::isfinite(weight.getData()[0]));
    EXPECT_TRUE(std::isfinite(weight.getData()[1]));
    EXPECT_NEAR(weight.getData()[0], -1e-20f, 1e-26f);
    EXPECT_FLOAT_EQ(weight.getData()[1], 1.0f);
}

} // namespace smart_dnn

#endif // TEST_ADADELTA_OPTIMIZER_CPP

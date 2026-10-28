#ifndef TEST_ADAGRAD_OPTIMIZER_CPP
#define TEST_ADAGRAD_OPTIMIZER_CPP

#include <cmath>
#include <gtest/gtest.h>
#include "../../smart_dnn/Optimizers/AdaGradOptimizer.hpp"

namespace smart_dnn {

TEST(AdaGradOptimizerTest, AccumulatesSquaredGradientsPerParameter) {
    AdaGradOptions<float> options;
    options.learningRate = 0.2f;
    options.epsilon = 0.1f;
    AdaGradOptimizer<float> optimizer(options);

    Tensor<float> weight({2, 1}, {1.0f, 2.0f});
    Tensor<float> gradient({2, 1}, {0.5f, 1.0f});
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};

    const float firstExpected[] = {
        1.0f - 0.2f * 0.5f / (0.5f + 0.1f),
        2.0f - 0.2f * 1.0f / (1.0f + 0.1f)
    };
    optimizer.optimize(weights, gradients);
    EXPECT_NEAR(weight.getData().data()[0], firstExpected[0], 1e-6f);
    EXPECT_NEAR(weight.getData().data()[1], firstExpected[1], 1e-6f);

    const float secondExpected[] = {
        firstExpected[0] - 0.2f * 0.5f / (std::sqrt(0.5f) + 0.1f),
        firstExpected[1] - 0.2f * 1.0f / (std::sqrt(2.0f) + 0.1f)
    };
    optimizer.optimize(weights, gradients);
    EXPECT_NEAR(weight.getData().data()[0], secondExpected[0], 1e-6f);
    EXPECT_NEAR(weight.getData().data()[1], secondExpected[1], 1e-6f);
}

TEST(AdaGradOptimizerTest, KeepsStateIsolatedBetweenWeightTensors) {
    AdaGradOptions<float> options;
    options.learningRate = 0.1f;
    AdaGradOptimizer<float> optimizer(options);

    Tensor<float> weightA({1, 1}, {1.0f});
    Tensor<float> weightB({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {1.0f});
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};
    std::vector<std::reference_wrapper<Tensor<float>>> weightsA = {std::ref(weightA)};
    std::vector<std::reference_wrapper<Tensor<float>>> weightsB = {std::ref(weightB)};

    optimizer.optimize(weightsA, gradients);
    const float firstUpdate = weightA.getData().data()[0];
    optimizer.optimize(weightsA, gradients);
    optimizer.optimize(weightsB, gradients);

    EXPECT_NEAR(weightB.getData().data()[0], firstUpdate, 1e-6f);
    EXPECT_NE(weightA.getData().data()[0], weightB.getData().data()[0]);
}

TEST(AdaGradOptimizerTest, AppliesPositiveLearningRateOverride) {
    AdaGradOptions<float> options;
    options.learningRate = 0.1f;
    options.epsilon = 0.1f;
    AdaGradOptimizer<float> optimizer(options);

    Tensor<float> weight({1, 1}, {1.0f});
    Tensor<float> gradient({1, 1}, {1.0f});
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};

    optimizer.optimize(weights, gradients, 0.3f);

    EXPECT_NEAR(weight.getData().data()[0], 1.0f - 0.3f / 1.1f, 1e-6f);
}

TEST(AdaGradOptimizerTest, RejectsMismatchedWeightAndGradientCounts) {
    AdaGradOptimizer<float> optimizer;
    Tensor<float> weight({1, 1}, {1.0f});
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients;

    EXPECT_THROW(optimizer.optimize(weights, gradients), std::invalid_argument);
}

TEST(AdaGradOptimizerTest, RemainsFiniteForTinyAndZeroGradients) {
    AdaGradOptions<float> options;
    options.learningRate = 0.1f;
    options.epsilon = 1e-8f;
    AdaGradOptimizer<float> optimizer(options);

    Tensor<float> weight({2, 1}, {0.0f, 2.0f});
    Tensor<float> gradient({2, 1}, {1e-10f, 0.0f});
    std::vector<std::reference_wrapper<Tensor<float>>> weights = {std::ref(weight)};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients = {std::ref(gradient)};

    optimizer.optimize(weights, gradients);

    EXPECT_TRUE(std::isfinite(weight.getData().data()[0]));
    EXPECT_GT(weight.getData().data()[0], 0.0f);
    EXPECT_EQ(weight.getData().data()[1], 2.0f);
}

} // namespace smart_dnn

#endif // TEST_ADAGRAD_OPTIMIZER_CPP

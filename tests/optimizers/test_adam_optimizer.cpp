#ifndef TEST_ADAM_OPTIMIZER_CPP
#define TEST_ADAM_OPTIMIZER_CPP

#include <gtest/gtest.h>
#include <cmath>
#include "../../smart_dnn/Optimizers/AdamOptimizer.hpp"

namespace smart_dnn {

template <typename T>
class AdamOptimizerTest : public ::testing::Test {};

using AdamPrecisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(AdamOptimizerTest, AdamPrecisions);

TYPED_TEST(AdamOptimizerTest, FirstStepMovesEachWeightByTheLearningRate) {
    using T = TypeParam;
    AdamOptions<T> options;
    options.learningRate = T(0.1);
    AdamOptimizer<T> optimizer(options);
    Tensor<T> weight({3}, {T(1), T(-2), T(3)});
    Tensor<T> gradient({3}, {T(0.5), T(-0.25), T(4)});

    optimizer.optimize({std::ref(weight)}, {std::ref(gradient)});

    // Bias-corrected moments make the first step learningRate * sign(gradient).
    EXPECT_NEAR(weight[0], 0.9, 1e-6);
    EXPECT_NEAR(weight[1], -1.9, 1e-6);
    EXPECT_NEAR(weight[2], 2.9, 1e-6);
}

TYPED_TEST(AdamOptimizerTest, FollowsTheAdamRecurrenceOverSeveralSteps) {
    using T = TypeParam;
    AdamOptions<T> options;
    options.learningRate = T(0.01);
    AdamOptimizer<T> optimizer(options);
    Tensor<T> weight({2}, {T(1), T(-2)});
    const T gradients[3][2] = {{T(0.5), T(-0.25)}, {T(-1), T(0.75)}, {T(0.2), T(0.1)}};
    T expected[2] = {T(1), T(-2)};
    T m[2] = {T(0), T(0)};
    T v[2] = {T(0), T(0)};

    for (int t = 1; t <= 3; ++t) {
        // The last step overrides the configured learning rate.
        const T rate = t == 3 ? T(0.02) : T(0.01);
        Tensor<T> gradient({2}, {gradients[t - 1][0], gradients[t - 1][1]});
        optimizer.optimize({std::ref(weight)}, {std::ref(gradient)}, t == 3 ? rate : T(-1));

        for (int i = 0; i < 2; ++i) {
            const T g = gradients[t - 1][i];
            m[i] = T(0.9) * m[i] + T(0.1) * g;
            v[i] = T(0.999) * v[i] + T(0.001) * g * g;
            const T mHat = m[i] / (T(1) - std::pow(T(0.9), t));
            const T vHat = v[i] / (T(1) - std::pow(T(0.999), t));
            expected[i] -= rate * mHat / (std::sqrt(vHat) + T(1e-8));
            EXPECT_NEAR(weight[i], expected[i], 1e-6) << "step " << t << ", weight " << i;
        }
    }
}

} // namespace smart_dnn

#endif // TEST_ADAM_OPTIMIZER_CPP

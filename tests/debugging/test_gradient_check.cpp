#ifndef TEST_GRADIENT_CHECK_CPP
#define TEST_GRADIENT_CHECK_CPP

#include <gtest/gtest.h>
#include <cmath>
#include "../../smart_dnn/Debugging/GradientCheck.hpp"

namespace smart_dnn {

namespace {

// sum(x^3), whose gradient is 3x^2.
float cubeSum(const Tensor<float>& x) {
    float sum = 0.0f;
    for (float value : x.getData()) {
        sum += value * value * value;
    }
    return sum;
}

} // namespace

TEST(GradientCheckTest, AcceptsACorrectGradient) {
    Tensor<float> x({2, 3}, {-1.5f, -0.5f, 0.0f, 0.25f, 1.0f, 2.0f});
    Tensor<float> gradient = x * x * 3.0f;

    GradientCheckResult<float> result = checkGradient(cubeSum, x, gradient);

    EXPECT_TRUE(result.passed) << "index " << result.index << ": analytic " << result.analytic << ", numeric " << result.numeric;
}

TEST(GradientCheckTest, ReportsTheWrongCoordinate) {
    Tensor<float> x({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> gradient({4}, {3.0f, 12.0f, 20.0f, 48.0f});  // Index 2 should be 27.

    GradientCheckResult<float> result = checkGradient(cubeSum, x, gradient);

    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.index, 2u);
    EXPECT_NEAR(result.numeric, 27.0f, 0.05f);
    EXPECT_FALSE(checkGradient([](const Tensor<float>&) { return std::nanf(""); }, x, x * x * 3.0f).passed);
}

TEST(GradientCheckTest, RejectsMismatchedShapesAndInvalidOptions) {
    Tensor<float> x({2}, {1.0f, 2.0f});

    EXPECT_THROW(checkGradient(cubeSum, x, Tensor<float>({1, 2}, 0.0f)), std::invalid_argument);
    EXPECT_THROW(checkGradient(cubeSum, x, x, {0.0f, 1e-2f}), std::invalid_argument);
}

} // namespace smart_dnn

#endif // TEST_GRADIENT_CHECK_CPP

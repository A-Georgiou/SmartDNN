#ifndef TEST_GELU_CPP
#define TEST_GELU_CPP

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "../../smart_dnn/Activations/GELU.hpp"
#include "../../smart_dnn/Debugging/GradientCheck.hpp"
#include "../../smart_dnn/Layers/ActivationLayer.hpp"

namespace smart_dnn {

TEST(GELUTest, MatchesTheTanhApproximation) {
    Tensor<float> output = GELU<float>().forward(Tensor<float>({4}, {-1.0f, 0.0f, 1.0f, 2.0f}));

    EXPECT_NEAR(output[0], -0.158808f, 1e-5);
    EXPECT_NEAR(output[1], 0.0f, 1e-5);
    EXPECT_NEAR(output[2], 0.841192f, 1e-5);
    EXPECT_NEAR(output[3], 1.954597f, 1e-5);
}

TEST(GELUTest, PassesGradientCheck) {
    ActivationLayer<float> gelu{GELU<float>()};
    Tensor<float> input({2, 5}, {-3.0f, -1.5f, -0.5f, -0.1f, 0.0f, 0.2f, 0.7f, 1.0f, 2.5f, 4.0f});
    Tensor<float> upstream({2, 5}, {1.0f, -0.5f, 2.0f, 0.3f, -1.2f, 0.8f, -2.0f, 1.5f, 0.6f, -0.4f});

    for (const GradientCheckResult<float>& result : checkLayerGradients(gelu, input, upstream)) {
        EXPECT_TRUE(result.passed) << "index " << result.index << ": analytic " << result.analytic << ", numeric " << result.numeric;
    }
}

TEST(GELUTest, KeepsRepresentableNegativeTails) {
    // References from the same formula in 60-digit decimal arithmetic.
    const Tensor<float> x({2}, {-6.0f, -10.0f});
    const Tensor<float> y = GELU<float>().forward(x), dx = GELU<float>().backward(x, Tensor<float>({2}, 1.0f));
    EXPECT_NEAR(y[0] / -8.4396452066938901e-11f, 1.0f, 1e-4f);
    EXPECT_NEAR(dx[0] / -7.7099725642306271e-10f, 1.0f, 1e-4f);
    EXPECT_NEAR(y[1] / -1.2040923319236964e-37f, 1.0f, 1e-4f);
    EXPECT_NEAR(dx[1] / -2.7576380161249452e-36f, 1.0f, 1e-4f);
    const Tensor<double> xd({1}, {-8.0});
    EXPECT_NEAR(GELU<double>().forward(xd)[0] / -3.1077829375011071e-21, 1.0, 1e-4);
    EXPECT_NEAR(GELU<double>().backward(xd, Tensor<double>({1}, 1.0))[0] / -4.7147845041068353e-20, 1.0, 1e-4);
}

TEST(GELUTest, SaturatesForHugeInputsAndKeepsNaN) {
    const float big = std::numeric_limits<float>::max();
    const Tensor<float> x({4}, {1e20f, big, -big, std::numeric_limits<float>::quiet_NaN()});
    const Tensor<float> y = GELU<float>().forward(x), dx = GELU<float>().backward(x, Tensor<float>({4}, -2.0f));
    EXPECT_EQ(y[0], 1e20f);
    EXPECT_EQ(y[1], big);
    EXPECT_EQ(y[2], 0.0f);
    EXPECT_TRUE(std::isnan(y[3]));
    EXPECT_EQ(dx[0], -2.0f);
    EXPECT_EQ(dx[1], -2.0f);
    EXPECT_EQ(dx[2], 0.0f);
    EXPECT_TRUE(std::isnan(dx[3]));
    EXPECT_EQ(GELU<double>().backward(Tensor<double>({1}, {1e155}), Tensor<double>({1}, 1.0))[0], 1.0);
}

} // namespace smart_dnn

#endif // TEST_GELU_CPP

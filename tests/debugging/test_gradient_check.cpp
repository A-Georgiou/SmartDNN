#ifndef TEST_GRADIENT_CHECK_CPP
#define TEST_GRADIENT_CHECK_CPP

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "../utils/tensor_helpers.hpp"
#include "../../smart_dnn/Debugging/GradientCheck.hpp"
#include "../../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include <vector>

namespace smart_dnn {

namespace {

// sum(x^3), whose gradient is 3x^2.
template <typename T>
T cubeSum(const Tensor<T>& x) {
    T sum = T(0);
    for (T value : x.getData()) {
        sum += value * value * value;
    }
    return sum;
}

} // namespace

template <typename T>
class GradientCheckTest : public ::testing::Test {};

using GradientCheckPrecisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GradientCheckTest, GradientCheckPrecisions);

TYPED_TEST(GradientCheckTest, AcceptsACorrectGradientAndLeavesInputsUnchanged) {
    using T = TypeParam;
    Tensor<T> x({2, 3}, {T(-1.5), T(-0.5), T(0), T(0.25), T(1), T(2)});
    Tensor<T> gradient = x * x * T(3);
    const Tensor<T> xBefore = x;
    const Tensor<T> gradientBefore = gradient;

    GradientCheckResult<T> result = checkGradient(cubeSum<T>, x, gradient);

    EXPECT_TRUE(result.passed) << "index " << result.index << ": analytic " << result.analytic << ", numeric " << result.numeric;
    EXPECT_EQ(result.checked, 6u);
    EXPECT_TRUE(TensorEquals(x, xBefore));
    EXPECT_TRUE(TensorEquals(gradient, gradientBefore));
}

TYPED_TEST(GradientCheckTest, FailsOnWrongOrNonFiniteGradients) {
    using T = TypeParam;
    Tensor<T> x({4}, {T(1), T(2), T(3), T(4)});
    Tensor<T> gradient({4}, {T(3), T(12), T(20), T(48)});  // Index 2 should be 27.

    GradientCheckResult<T> result = checkGradient(cubeSum<T>, x, gradient);

    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.index, 2u);
    EXPECT_NEAR(result.numeric, 27.0, 0.05);
    EXPECT_NEAR(result.maxAbsoluteError, 7.0, 0.05);

    Tensor<T> nanGradient({4}, {T(3), T(12), std::numeric_limits<T>::quiet_NaN(), T(48)});
    EXPECT_FALSE(checkGradient(cubeSum<T>, x, nanGradient).passed);
    EXPECT_FALSE(checkGradient([](const Tensor<T>&) { return std::numeric_limits<T>::quiet_NaN(); }, x, x * x * T(3)).passed);
    // At 1e8 a float step of 1e-3 rounds away, which must fail rather than accept a zero gradient.
    EXPECT_FALSE(checkGradient([](const Tensor<T>& p) { return p[0]; }, Tensor<T>({1}, {T(1e8)}), Tensor<T>({1}, {T(0)})).passed);
}

TYPED_TEST(GradientCheckTest, RejectsMismatchedShapesAndInvalidOptions) {
    using T = TypeParam;
    Tensor<T> x({2}, {T(1), T(2)});

    EXPECT_THROW(checkGradient(cubeSum<T>, x, Tensor<T>({1, 2}, T(0))), std::invalid_argument);
    EXPECT_THROW(checkGradient(cubeSum<T>, x, x, {T(0), T(1e-2)}), std::invalid_argument);
    EXPECT_THROW(checkGradient(cubeSum<T>, x, x, {T(1e-3), std::numeric_limits<T>::infinity()}), std::invalid_argument);
}

namespace {

// Doubles its input but reports three times the upstream gradient.
class WrongBackwardLayer : public Layer<float> {
public:
    Tensor<float> forward(const Tensor<float>& input) override { return input * 2.0f; }
    Tensor<float> backward(const Tensor<float>& gradOutput) override { return gradOutput * 3.0f; }
};

// Throws once its weight is perturbed, so a parameter check stops partway through.
class ThrowsWhenPerturbedLayer : public Layer<float> {
public:
    Tensor<float> forward(const Tensor<float>& input) override {
        if (weight[0] != 1.0f) {
            throw std::runtime_error("perturbed");
        }
        return input;
    }
    Tensor<float> backward(const Tensor<float>& gradOutput) override { return gradOutput; }
    std::vector<Parameter<float>> parameters() override { return {{weight, gradient}}; }

    Tensor<float> weight{Shape({1}), 1.0f};
    Tensor<float> gradient{Shape({1}), 0.0f};
};

} // namespace

TEST(CheckLayerGradientsTest, PassesForFullyConnectedInputAndParameters) {
    FullyConnectedLayer<float> layer(3, 2);
    const Tensor<float> weightsBefore = layer.getWeights();
    Tensor<float> input({2, 3}, {0.5f, -1.0f, 2.0f, 1.5f, 0.25f, -0.75f});
    Tensor<float> upstream({2, 2}, {1.0f, -2.0f, 0.5f, 3.0f});

    std::vector<GradientCheckResult<float>> results = checkLayerGradients(layer, input, upstream);

    ASSERT_EQ(results.size(), 3u);
    for (const GradientCheckResult<float>& result : results) {
        EXPECT_TRUE(result.passed) << "index " << result.index << ": analytic " << result.analytic << ", numeric " << result.numeric;
    }
    EXPECT_TRUE(TensorEquals(layer.getWeights(), weightsBefore));
}

TEST(CheckLayerGradientsTest, RestoresParametersWhenTheObjectiveThrows) {
    ThrowsWhenPerturbedLayer layer;

    EXPECT_THROW(checkLayerGradients(layer, Tensor<float>({1}, 1.0f), Tensor<float>({1}, 1.0f), false), std::runtime_error);
    EXPECT_EQ(layer.weight[0], 1.0f);
}

TEST(CheckLayerGradientsTest, FailsForAWrongBackward) {
    WrongBackwardLayer layer;

    std::vector<GradientCheckResult<float>> results = checkLayerGradients(layer, Tensor<float>({2}, {1.0f, -1.0f}), Tensor<float>({2}, 1.0f));

    ASSERT_EQ(results.size(), 1u);
    EXPECT_FALSE(results[0].passed);
}

} // namespace smart_dnn

#endif // TEST_GRADIENT_CHECK_CPP

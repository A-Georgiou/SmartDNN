#ifndef TEST_GELU_CPP
#define TEST_GELU_CPP

#include <gtest/gtest.h>
#include <cmath>
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

} // namespace smart_dnn

#endif // TEST_GELU_CPP

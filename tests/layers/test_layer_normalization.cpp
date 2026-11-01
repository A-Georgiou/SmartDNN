#ifndef TEST_LAYER_NORMALIZATION_CPP
#define TEST_LAYER_NORMALIZATION_CPP

#include <gtest/gtest.h>
#include <cmath>
#include "../utils/tensor_helpers.hpp"
#include "../../smart_dnn/Debugging/GradientCheck.hpp"
#include "../../smart_dnn/Regularisation/LayerNormalizationLayer.hpp"

namespace smart_dnn {

TEST(LayerNormalizationLayerTest, NormalisesEachRowOfTheLastAxis) {
    LayerNormalizationLayer<float> norm(4);

    Tensor<float> output = norm.forward(Tensor<float>({2, 4}, {1, 2, 3, 4, -2, 0, 2, 8}));

    for (int row = 0; row < 2; ++row) {
        float mean = 0.0f;
        float variance = 0.0f;
        for (int j = 0; j < 4; ++j) {
            mean += output[row * 4 + j] / 4.0f;
        }
        for (int j = 0; j < 4; ++j) {
            variance += (output[row * 4 + j] - mean) * (output[row * 4 + j] - mean) / 4.0f;
        }
        EXPECT_NEAR(mean, 0.0f, 1e-6);
        EXPECT_NEAR(variance, 1.0f, 1e-4);
    }
}

TEST(LayerNormalizationLayerTest, PassesGradientCheckWithScaleAndShift) {
    LayerNormalizationLayer<float> norm(5);
    norm.parameters()[0].value.get() = Tensor<float>({1, 5}, {1.5f, -0.5f, 2.0f, 0.75f, 1.0f});
    norm.parameters()[1].value.get() = Tensor<float>({1, 5}, {0.1f, 0.2f, -0.3f, 0.0f, 0.5f});
    Tensor<float> input({2, 3, 5}, 0.0f);
    Tensor<float> upstream({2, 3, 5}, 0.0f);
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        input[i] = std::sin(0.7f * static_cast<float>(i));
        upstream[i] = std::cos(1.3f * static_cast<float>(i));
    }

    for (const GradientCheckResult<float>& result : checkLayerGradients(norm, input, upstream)) {
        EXPECT_TRUE(result.passed) << "index " << result.index << ": analytic " << result.analytic << ", numeric " << result.numeric;
    }
}

} // namespace smart_dnn

#endif // TEST_LAYER_NORMALIZATION_CPP

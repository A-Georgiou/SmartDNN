#ifndef TEST_TRANSFORMER_BLOCK_CPP
#define TEST_TRANSFORMER_BLOCK_CPP

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include "../utils/tensor_helpers.hpp"
#include "../../smart_dnn/Debugging/GradientCheck.hpp"
#include "../../smart_dnn/Layers/TransformerBlock.hpp"

namespace smart_dnn {

namespace {

Tensor<float> wave(const Shape& shape, float frequency, float phase) {
    Tensor<float> tensor(shape, 0.0f);
    for (size_t i = 0; i < shape.size(); ++i) {
        tensor[i] = std::sin(frequency * static_cast<float>(i) + phase);
    }
    return tensor;
}

} // namespace

TEST(TransformerBlockTest, PassesGradientCheck) {
    TransformerBlock<float> block(8, 2);

    for (const GradientCheckResult<float>& result : checkLayerGradients(block, wave(Shape({2, 3, 8}), 0.7f, 0.0f), wave(Shape({2, 3, 8}), 1.3f, 0.5f))) {
        EXPECT_TRUE(result.passed) << "index " << result.index << ": analytic " << result.analytic << ", numeric " << result.numeric;
    }
}

TEST(TransformerBlockTest, SaveAndLoadReproduceTheOutput) {
    TransformerBlock<float> trained(8, 2);
    TransformerBlock<float> restored(8, 2);
    const Tensor<float> input = wave(Shape({1, 3, 8}), 0.7f, 0.0f);
    std::stringstream stream;

    trained.save(stream);
    restored.load(stream);

    EXPECT_TRUE(TensorEquals(restored.forward(input), trained.forward(input)));
}

} // namespace smart_dnn

#endif // TEST_TRANSFORMER_BLOCK_CPP

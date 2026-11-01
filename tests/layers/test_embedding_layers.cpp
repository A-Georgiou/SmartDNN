#ifndef TEST_EMBEDDING_LAYERS_CPP
#define TEST_EMBEDDING_LAYERS_CPP

#include <gtest/gtest.h>
#include <cmath>
#include "../utils/tensor_helpers.hpp"
#include "../../smart_dnn/Debugging/GradientCheck.hpp"
#include "../../smart_dnn/Layers/EmbeddingLayer.hpp"
#include "../../smart_dnn/Layers/PositionalEmbeddingLayer.hpp"

namespace smart_dnn {

namespace {

Tensor<float> wave(const Shape& shape, float phase) {
    Tensor<float> tensor(shape, 0.0f);
    for (size_t i = 0; i < shape.size(); ++i) {
        tensor[i] = std::sin(0.7f * static_cast<float>(i) + phase);
    }
    return tensor;
}

} // namespace

TEST(EmbeddingLayerTest, LooksUpRowsAndAccumulatesRepeatedIds) {
    EmbeddingLayer<float> embedding(4, 2);
    embedding.parameters()[0].value.get() = Tensor<float>({4, 2}, {0, 1, 10, 11, 20, 21, 30, 31});

    Tensor<float> output = embedding.forward(Tensor<float>({2, 2}, {1, 3, 1, 0}));
    embedding.backward(Tensor<float>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));

    EXPECT_TRUE(TensorEquals(output, Tensor<float>({2, 2, 2}, {10, 11, 30, 31, 10, 11, 0, 1})));
    EXPECT_TRUE(TensorEquals(embedding.parameters()[0].gradient.get(), Tensor<float>({4, 2}, {7, 8, 6, 8, 0, 0, 3, 4})));
}

TEST(EmbeddingLayerTest, RejectsIdsOutsideTheVocabulary) {
    EmbeddingLayer<float> embedding(4, 2);

    EXPECT_THROW(embedding.forward(Tensor<float>({2}, {0, 4})), std::invalid_argument);
    EXPECT_THROW(embedding.forward(Tensor<float>({2}, {0, 1.5f})), std::invalid_argument);
}

TEST(PositionalEmbeddingLayerTest, PassesGradientCheckOnShorterSequences) {
    PositionalEmbeddingLayer<float> positions(5, 3);

    for (const GradientCheckResult<float>& result : checkLayerGradients(positions, wave(Shape({2, 4, 3}), 0.0f), wave(Shape({2, 4, 3}), 1.0f))) {
        EXPECT_TRUE(result.passed) << "index " << result.index << ": analytic " << result.analytic << ", numeric " << result.numeric;
    }
}

TEST(PositionalEmbeddingLayerTest, RejectsSequencesLongerThanMaxLength) {
    PositionalEmbeddingLayer<float> positions(5, 3);

    EXPECT_THROW(positions.forward(Tensor<float>({1, 6, 3}, 0.0f)), std::invalid_argument);
}

} // namespace smart_dnn

#endif // TEST_EMBEDDING_LAYERS_CPP

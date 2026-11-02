#ifndef TEST_KV_CACHE_DECODER_CPP
#define TEST_KV_CACHE_DECODER_CPP

#include <gtest/gtest.h>
#include <vector>
#include "../../smart_dnn/Inference/KVCacheDecoder.hpp"

namespace smart_dnn {

namespace {

void addTinyGpt(SmartDNN<float>& model) {
    model.addLayer(EmbeddingLayer<float>(7, 8));
    model.addLayer(PositionalEmbeddingLayer<float>(6, 8));
    model.addLayer(TransformerBlock<float>(8, 2));
    model.addLayer(TransformerBlock<float>(8, 2));
    model.addLayer(LayerNormalizationLayer<float>(8));
    model.addLayer(FullyConnectedLayer<float>(8, 7));
}

} // namespace

TEST(KVCacheDecoderTest, MatchesTheFullForwardPassAtEveryPosition) {
    SmartDNN<float> model;
    addTinyGpt(model);
    const std::vector<int> tokens = {3, 0, 6, 2, 2, 5};
    Tensor<float> sequence(Shape({1, 6}), 0.0f);
    for (size_t t = 0; t < tokens.size(); ++t) {
        sequence[t] = static_cast<float>(tokens[t]);
    }
    const Tensor<float> expected = model.predict(sequence);
    KVCacheDecoder<float> decoder(model);

    for (size_t t = 0; t < tokens.size(); ++t) {
        const std::vector<float> logits = decoder.push(tokens[t]);
        ASSERT_EQ(logits.size(), 7u);
        for (size_t v = 0; v < logits.size(); ++v) {
            EXPECT_NEAR(logits[v], expected[t * 7 + v], 1e-5) << "position " << t << ", class " << v;
        }
    }
}

TEST(KVCacheDecoderTest, RefusesTokensBeyondItsCapacityUntilReset) {
    SmartDNN<float> model;
    addTinyGpt(model);
    KVCacheDecoder<float> decoder(model);
    for (int t = 0; t < 6; ++t) {
        decoder.push(t);
    }

    EXPECT_THROW(decoder.push(1), std::length_error);
    decoder.reset();
    EXPECT_EQ(decoder.size(), 0u);
    EXPECT_NO_THROW(decoder.push(1));
}

TEST(KVCacheDecoderTest, RejectsModelsItCannotDecode) {
    SmartDNN<float> notGpt;
    notGpt.addLayer(FullyConnectedLayer<float>(4, 4));
    SmartDNN<float> notCausal;
    notCausal.addLayer(EmbeddingLayer<float>(7, 8));
    notCausal.addLayer(PositionalEmbeddingLayer<float>(6, 8));
    notCausal.addLayer(TransformerBlock<float>(8, 2, false));
    notCausal.addLayer(LayerNormalizationLayer<float>(8));
    notCausal.addLayer(FullyConnectedLayer<float>(8, 7));
    SmartDNN<float> narrowEmbedding;
    narrowEmbedding.addLayer(EmbeddingLayer<float>(7, 4));
    narrowEmbedding.addLayer(PositionalEmbeddingLayer<float>(6, 8));
    narrowEmbedding.addLayer(TransformerBlock<float>(8, 2));
    narrowEmbedding.addLayer(LayerNormalizationLayer<float>(8));
    narrowEmbedding.addLayer(FullyConnectedLayer<float>(8, 7));

    EXPECT_THROW(KVCacheDecoder<float>{notGpt}, std::invalid_argument);
    EXPECT_THROW(KVCacheDecoder<float>{notCausal}, std::invalid_argument);
    EXPECT_THROW(KVCacheDecoder<float>{narrowEmbedding}, std::invalid_argument);
}

} // namespace smart_dnn

#endif // TEST_KV_CACHE_DECODER_CPP

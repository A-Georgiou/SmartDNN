#ifndef TEST_MODEL_TRAINING_CPP
#define TEST_MODEL_TRAINING_CPP

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>
#include "../utils/tensor_helpers.hpp"
#include "../../smart_dnn/SmartDNN.hpp"
#include "../../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../../smart_dnn/Layers/Conv2DLayer.hpp"
#include "../../smart_dnn/Layers/EmbeddingLayer.hpp"
#include "../../smart_dnn/Layers/PositionalEmbeddingLayer.hpp"
#include "../../smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "../../smart_dnn/Regularisation/LayerNormalizationLayer.hpp"
#include "../../smart_dnn/Layers/ActivationLayer.hpp"
#include "../../smart_dnn/Activations/Tanh.hpp"
#include "../../smart_dnn/Loss/MSELoss.hpp"

namespace smart_dnn {

namespace {

struct OptimizerLog {
    int calls = 0;
    size_t tensors = 0;
    float learningRate = 0.0f;
    std::vector<Tensor<float>> gradients;
};

// Records each optimizer call and leaves the weights unchanged.
class LoggingOptimizer : public Optimizer<float> {
public:
    explicit LoggingOptimizer(OptimizerLog& log) : log(log) {}

    void optimize(const std::vector<std::reference_wrapper<Tensor<float>>>& weights,
                  const std::vector<std::reference_wrapper<Tensor<float>>>& gradients,
                  float learningRateOverride = -1.0f) override {
        ++log.calls;
        log.tensors = weights.size();
        log.learningRate = learningRateOverride;
        log.gradients.clear();
        for (const Tensor<float>& gradient : gradients) {
            log.gradients.push_back(gradient);
        }
    }

private:
    OptimizerLog& log;
};

const Tensor<float> input({2, 3}, {0.5f, -1.0f, 2.0f, 1.5f, 0.25f, -0.75f});

} // namespace

TEST(SmartDNNTrainingTest, UpdateWeightsMakesOneOptimizerCallForAllLayers) {
    OptimizerLog log;
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer<float>(3, 4));
    model.addLayer(ActivationLayer<float>(Tanh<float>()));
    model.addLayer(FullyConnectedLayer<float>(4, 2));
    model.compile(MSELoss<float>(), LoggingOptimizer(log));

    model.backward(Tensor<float>(model.predict(input).getShape(), 1.0f));
    model.updateWeights();

    EXPECT_EQ(log.calls, 1);
    EXPECT_EQ(log.tensors, 4u);
}

TEST(SmartDNNTrainingTest, TrainStepReturnsTheLossAndPassesTheLearningRate) {
    OptimizerLog log;
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer<float>(3, 2));
    model.compile(MSELoss<float>(), LoggingOptimizer(log));
    Tensor<float> target({2, 2}, {1.0f, 0.0f, -1.0f, 0.5f});

    float loss = model.trainStep(input, target, 0.05f);

    EXPECT_FLOAT_EQ(loss, MSELoss<float>().compute(model.predict(input), target)[0]);
    EXPECT_EQ(log.calls, 1);
    EXPECT_FLOAT_EQ(log.learningRate, 0.05f);
}

TEST(SmartDNNTrainingTest, MaxGradientNormScalesAllGradientsTogether) {
    OptimizerLog log;
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer<float>(3, 2));
    model.compile(MSELoss<float>(), LoggingOptimizer(log));
    Tensor<float> target({2, 2}, {10.0f, -10.0f, 10.0f, -10.0f});

    model.trainStep(input, target);
    std::vector<Tensor<float>> unclipped = log.gradients;
    float squaredNorm = 0.0f;
    for (const Tensor<float>& gradient : unclipped) {
        for (float value : gradient.getData()) {
            squaredNorm += value * value;
        }
    }
    ASSERT_GT(std::sqrt(squaredNorm), 1.0f);

    model.setMaxGradientNorm(1.0f);
    model.trainStep(input, target);

    ASSERT_EQ(log.gradients.size(), unclipped.size());
    for (size_t i = 0; i < unclipped.size(); ++i) {
        EXPECT_TRUE(TensorEquals(log.gradients[i], unclipped[i] * (1.0f / std::sqrt(squaredNorm))));
    }
}

TEST(SmartDNNTrainingTest, MaxGradientNormClipsHugeGradientsToTheLimit) {
    OptimizerLog log;
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer<float>(3, 2));
    model.compile(MSELoss<float>(), LoggingOptimizer(log));
    model.setMaxGradientNorm(1.0f);

    // Gradients near 1e20 overflow float when squared directly.
    model.trainStep(input, Tensor<float>({2, 2}, 1e20f));

    double squaredNorm = 0.0;
    for (const Tensor<float>& gradient : log.gradients) {
        for (float value : gradient.getData()) {
            squaredNorm += static_cast<double>(value) * value;
        }
    }
    EXPECT_NEAR(std::sqrt(squaredNorm), 1.0, 1e-5);
}

TEST(SmartDNNTrainingTest, SetMaxGradientNormRejectsInvalidLimits) {
    SmartDNN<float> model;

    EXPECT_THROW(model.setMaxGradientNorm(-1.0f), std::invalid_argument);
    EXPECT_THROW(model.setMaxGradientNorm(std::nanf("")), std::invalid_argument);
    EXPECT_THROW(model.setMaxGradientNorm(std::numeric_limits<float>::infinity()), std::invalid_argument);
}

TEST(SmartDNNTrainingTest, GradientsAreOnlyAllocatedByBackward) {
    FullyConnectedLayer<float> dense(3, 2);
    Conv2DLayer<float> conv(1, 2, 3);
    BatchNormalizationLayer<float> batchNorm(2);
    LayerNormalizationLayer<float> layerNorm(2);
    EmbeddingLayer<float> embedding(4, 2);
    PositionalEmbeddingLayer<float> positions(3, 2);
    for (Layer<float>* layer : std::initializer_list<Layer<float>*>{&dense, &conv, &batchNorm, &layerNorm, &embedding, &positions}) {
        for (const Parameter<float>& parameter : layer->parameters()) {
            EXPECT_FALSE(parameter.hasGradient());
        }
    }

    OptimizerLog log;
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer<float>(3, 2));
    model.compile(MSELoss<float>(), LoggingOptimizer(log));
    EXPECT_THROW(model.updateWeights(), std::logic_error);
    LoggingOptimizer optimizer(log);
    EXPECT_THROW(dense.updateWeights(optimizer), std::logic_error);
    EXPECT_EQ(log.calls, 0);

    dense.forward(input);
    dense.backward(Tensor<float>({2, 2}, 1.0f));
    dense.updateWeights(optimizer);
    EXPECT_EQ(log.calls, 1);
}

} // namespace smart_dnn

#endif // TEST_MODEL_TRAINING_CPP

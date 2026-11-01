#include <gtest/gtest.h>
#include <vector>
#include "../../smart_dnn/Training/DataParallelTrainer.hpp"
#include "../../smart_dnn/Activations/Tanh.hpp"
#include "../../smart_dnn/Layers/ActivationLayer.hpp"
#include "../../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../../smart_dnn/Loss/MSELoss.hpp"
#include "../../smart_dnn/Optimizers/AdamOptimizer.hpp"

namespace smart_dnn {

namespace {

void addLayers(SmartDNN<float>& model) {
    model.addLayer(FullyConnectedLayer<float>(3, 4));
    model.addLayer(ActivationLayer<float>(Tanh<float>()));
    model.addLayer(FullyConnectedLayer<float>(4, 2));
}

} // namespace

TEST(DataParallelTrainerTest, MatchesSingleThreadTraining) {
    RandomEngine::setSeed(7);
    SmartDNN<float> serial;
    addLayers(serial);
    serial.compile(MSELoss<float>(), AdamOptimizer<float>());
    RandomEngine::setSeed(7);
    SmartDNN<float> parallel;
    addLayers(parallel);
    parallel.compile(MSELoss<float>(), AdamOptimizer<float>());
    DataParallelTrainer<float> trainer(parallel, addLayers, MSELoss<float>(), 3);

    const Tensor<float> inputs({5, 3}, {0.5f, -1.0f, 2.0f, 1.5f, 0.25f, -0.75f, -2.0f, 0.5f, 1.0f, 0.0f, 1.0f, -1.0f, 0.75f, 0.75f, -0.5f});
    const Tensor<float> targets({5, 2}, {1.0f, 0.0f, -1.0f, 0.5f, 0.25f, 0.25f, 0.0f, -0.5f, 1.0f, 1.0f});
    for (int step = 0; step < 3; ++step) {
        EXPECT_NEAR(trainer.trainStep(inputs, targets, 0.05f), serial.trainStep(inputs, targets, 0.05f), 1e-5f);
    }
    for (size_t layer = 0; layer < serial.getLayerCount(); ++layer) {
        const std::vector<Parameter<float>> expected = serial.getLayer(layer)->parameters();
        const std::vector<Parameter<float>> actual = parallel.getLayer(layer)->parameters();
        for (size_t p = 0; p < expected.size(); ++p) {
            for (size_t i = 0; i < expected[p].value.get().getShape().size(); ++i) {
                EXPECT_NEAR(actual[p].value.get()[i], expected[p].value.get()[i], 1e-5f);
            }
        }
    }
}

TEST(DataParallelTrainerTest, RejectsUnsupportedSetups) {
    SmartDNN<float> model;
    addLayers(model);
    model.compile(MSELoss<float>(), AdamOptimizer<float>());
    EXPECT_THROW(DataParallelTrainer<float>(model, addLayers, MSELoss<float>(), 0), std::invalid_argument);
    EXPECT_THROW(DataParallelTrainer<float>(model, [](SmartDNN<float>& copy) { copy.addLayer(FullyConnectedLayer<float>(3, 2)); }, MSELoss<float>(), 2),
                 std::invalid_argument);

    SmartDNN<float> withDropout;
    withDropout.addLayer(DropoutLayer<float>(0.5f));
    EXPECT_THROW(DataParallelTrainer<float>(withDropout, [](SmartDNN<float>&) {}, MSELoss<float>(), 2), std::invalid_argument);
}

} // namespace smart_dnn

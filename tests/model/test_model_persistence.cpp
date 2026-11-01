#ifndef TEST_MODEL_PERSISTENCE_CPP
#define TEST_MODEL_PERSISTENCE_CPP

#include <gtest/gtest.h>
#include <cstdio>
#include <filesystem>
#include "../utils/tensor_helpers.hpp"
#include "../../smart_dnn/SmartDNN.hpp"
#include "../../smart_dnn/Layers/Conv2DLayer.hpp"
#include "../../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../../smart_dnn/Layers/ActivationLayer.hpp"
#include "../../smart_dnn/Layers/FlattenLayer.hpp"
#include "../../smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "../../smart_dnn/Activations/ReLU.hpp"
#include "../../smart_dnn/Activations/Softmax.hpp"

namespace smart_dnn {

// Conv, BatchNorm and FC layers cover every kind of parameter saveModel writes.
void addPersistenceTestLayers(SmartDNN<float>& model, int outputs) {
    model.addLayer(Conv2DLayer(1, 2, 3));
    model.addLayer(BatchNormalizationLayer(2));
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(FlattenLayer());
    model.addLayer(FullyConnectedLayer(2 * 3 * 3, outputs));
    model.addLayer(ActivationLayer(Softmax()));
}

TEST(SmartDNNTest, SaveModelThenLoadModelReproducesPredictions) {
    const std::string path = (std::filesystem::temp_directory_path() / "smartdnn_save_load_test.bin").string();
    std::vector<float> pixels(2 * 5 * 5);
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = 0.1f * static_cast<float>(i % 7);
    const Tensor<float> input(Shape({2, 1, 5, 5}), pixels.data());

    SmartDNN<float> trained;
    addPersistenceTestLayers(trained, 4);
    trained.predict(input);  // A training-mode pass moves the BatchNorm running statistics off their defaults.
    trained.evalMode();
    const Tensor<float> expected = trained.predict(input);
    trained.saveModel(path);

    SmartDNN<float> restored;
    addPersistenceTestLayers(restored, 4);
    restored.loadModel(path);
    restored.evalMode();
    EXPECT_TRUE(TensorEquals(restored.predict(input), expected));

    SmartDNN<float> otherShape;
    addPersistenceTestLayers(otherShape, 5);
    EXPECT_THROW(otherShape.loadModel(path), std::runtime_error);
    std::remove(path.c_str());
}

} // namespace smart_dnn

#endif // TEST_MODEL_PERSISTENCE_CPP

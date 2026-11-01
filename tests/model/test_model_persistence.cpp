#ifndef TEST_MODEL_PERSISTENCE_CPP
#define TEST_MODEL_PERSISTENCE_CPP

#include <gtest/gtest.h>
#include <cmath>
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


TEST(SmartDNNTest, Int8ModelFilesAreUnderAThirdOfTheSizeAndPredictClosely) {
    const std::string floatPath = (std::filesystem::temp_directory_path() / "smartdnn_float_test.bin").string();
    const std::string int8Path = (std::filesystem::temp_directory_path() / "smartdnn_int8_test.bin").string();
    auto addLayers = [](SmartDNN<float>& model) {
        model.addLayer(FullyConnectedLayer(64, 128));
        model.addLayer(ActivationLayer(ReLU()));
        model.addLayer(FullyConnectedLayer(128, 10));
    };
    std::vector<float> values(4 * 64);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(0.37f * static_cast<float>(i));
    const Tensor<float> input(Shape({4, 64}), values.data());
    SmartDNN<float> trained;
    addLayers(trained);
    const Tensor<float> expected = trained.predict(input);

    trained.saveModel(floatPath);
    trained.saveModel(int8Path, WeightFormat::Int8);
    SmartDNN<float> restored;
    addLayers(restored);
    restored.loadModel(int8Path);

    EXPECT_LT(std::filesystem::file_size(int8Path) * 3, std::filesystem::file_size(floatPath));
    const Tensor<float> output = restored.predict(input);
    float largest = 0.0f;
    float worst = 0.0f;
    for (size_t i = 0; i < expected.getShape().size(); ++i) {
        largest = std::max(largest, std::abs(expected[i]));
        worst = std::max(worst, std::abs(output[i] - expected[i]));
    }
    EXPECT_LT(worst, 0.02f * largest);
    std::remove(floatPath.c_str());
    std::remove(int8Path.c_str());
}

} // namespace smart_dnn

#endif // TEST_MODEL_PERSISTENCE_CPP

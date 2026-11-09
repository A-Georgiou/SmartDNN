#ifndef TEST_DATA_AUGMENTATION_CPP
#define TEST_DATA_AUGMENTATION_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Layers/DataAugmentationLayer.hpp"
#include "../utils/tensor_helpers.hpp"

namespace smart_dnn {

TEST(DataAugmentationLayerTest, HorizontalFlipForwardAndBackward) {
    RandomFlip2DLayer<float> flipLayer(true, false, 1.0f);
    Tensor<float> input({1, 1, 2, 3}, {1.0f, 2.0f, 3.0f,
                                        4.0f, 5.0f, 6.0f});

    Tensor<float> output = flipLayer.forward(input);

    ValidateTensorShape(output, 4, 6, {1, 1, 2, 3});
    ValidateTensorData(output, {3.0f, 2.0f, 1.0f,
                                6.0f, 5.0f, 4.0f});

    Tensor<float> gradOutput({1, 1, 2, 3}, {10.0f, 20.0f, 30.0f,
                                             40.0f, 50.0f, 60.0f});
    Tensor<float> gradInput = flipLayer.backward(gradOutput);

    ValidateTensorData(gradInput, {30.0f, 20.0f, 10.0f,
                                   60.0f, 50.0f, 40.0f});
}

TEST(DataAugmentationLayerTest, Rotation90ForwardAndBackward) {
    RandomRotation90Layer<float> rotationLayer(1);
    Tensor<float> input({1, 1, 2, 3}, {1.0f, 2.0f, 3.0f,
                                        4.0f, 5.0f, 6.0f});

    Tensor<float> output = rotationLayer.forward(input);

    ValidateTensorShape(output, 4, 6, {1, 1, 3, 2});
    ValidateTensorData(output, {4.0f, 1.0f,
                                5.0f, 2.0f,
                                6.0f, 3.0f});

    Tensor<float> gradOutput({1, 1, 3, 2}, {10.0f, 20.0f,
                                             30.0f, 40.0f,
                                             50.0f, 60.0f});
    Tensor<float> gradInput = rotationLayer.backward(gradOutput);

    ValidateTensorShape(gradInput, 4, 6, {1, 1, 2, 3});
    ValidateTensorData(gradInput, {20.0f, 40.0f, 60.0f,
                                   10.0f, 30.0f, 50.0f});
}

TEST(DataAugmentationLayerTest, RandomCropForwardAndBackward) {
    RandomCrop2DLayer<float> cropLayer(2, 2, 1, 1);
    Tensor<float> input({1, 1, 3, 4}, {1.0f, 2.0f, 3.0f, 4.0f,
                                        5.0f, 6.0f, 7.0f, 8.0f,
                                        9.0f, 10.0f, 11.0f, 12.0f});

    Tensor<float> output = cropLayer.forward(input);

    ValidateTensorShape(output, 4, 4, {1, 1, 2, 2});
    ValidateTensorData(output, {6.0f, 7.0f,
                                10.0f, 11.0f});

    Tensor<float> gradOutput({1, 1, 2, 2}, {1.0f, 2.0f,
                                             3.0f, 4.0f});
    Tensor<float> gradInput = cropLayer.backward(gradOutput);

    ValidateTensorShape(gradInput, 4, 12, {1, 1, 3, 4});
    ValidateTensorData(gradInput, {0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 1.0f, 2.0f, 0.0f,
                                   0.0f, 3.0f, 4.0f, 0.0f});
}

TEST(DataAugmentationLayerTest, TrainingModeDisabledLeavesInputUnchanged) {
    RandomFlip2DLayer<float> flipLayer(true, true, 1.0f);
    flipLayer.setTrainingMode(false);
    Tensor<float> input({1, 1, 2, 2}, {1.0f, 2.0f,
                                        3.0f, 4.0f});

    Tensor<float> output = flipLayer.forward(input);

    ASSERT_TRUE(TensorEquals(input, output));
}

} // namespace smart_dnn

#endif // TEST_DATA_AUGMENTATION_CPP

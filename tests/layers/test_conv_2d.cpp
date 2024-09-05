
#ifndef TEST_CONV_2D_CPP
#define TEST_CONV_2D_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Layers/Conv2DLayer.hpp"
#include "../utils/tensor_helpers.hpp"
#include "../../smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "../../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../../smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "../../smart_dnn/Regularisation/DropoutLayer.hpp"

namespace smart_dnn {

TEST(Conv2DLayerTest, ForwardPassHardcodedConv2D) {
    Conv2DLayer<float> convLayer(1, 1, 3); // 1 input channel, 1 output channel, 3x3 kernel

    // Set weights and biases
    Tensor<float> weightValues({1, 1, 3, 3}, {1, 0, -1,
                                              0, 1, 0,
                                              -1, 0, 1});
    Tensor<float> biasValues({1, 1}, {0.1f});

    convLayer.setWeights(weightValues);
    convLayer.setBiases(biasValues);

    // Create an input tensor (1 batch, 1 channel, 5x5 image)
    Tensor<float> input({1, 1, 5, 5}, {1, 2, 3, 4, 5,
                                       2, 3, 4, 5, 6,
                                       3, 4, 5, 6, 7,
                                       4, 5, 6, 7, 8,
                                       5, 6, 7, 8, 9});

    // Perform forward pass
    Tensor<float> output = convLayer.forward(input);

    // Expected output shape: (1, 1, 3, 3)
    ASSERT_EQ(output.getShape(), Shape({1, 1, 3, 3}));

    // Calculate expected output manually
    std::vector<float> expectedOutput = {
        1*1 + 2*0 + 3*(-1) + 2*0 + 3*1 + 4*0 + 3*(-1) + 4*0 + 5*1 + 0.1f,
        2*1 + 3*0 + 4*(-1) + 3*0 + 4*1 + 5*0 + 4*(-1) + 5*0 + 6*1 + 0.1f,
        3*1 + 4*0 + 5*(-1) + 4*0 + 5*1 + 6*0 + 5*(-1) + 6*0 + 7*1 + 0.1f,
        2*1 + 3*0 + 4*(-1) + 3*0 + 4*1 + 5*0 + 4*(-1) + 5*0 + 6*1 + 0.1f,
        3*1 + 4*0 + 5*(-1) + 4*0 + 5*1 + 6*0 + 5*(-1) + 6*0 + 7*1 + 0.1f,
        4*1 + 5*0 + 6*(-1) + 5*0 + 6*1 + 7*0 + 6*(-1) + 7*0 + 8*1 + 0.1f,
        3*1 + 4*0 + 5*(-1) + 4*0 + 5*1 + 6*0 + 5*(-1) + 6*0 + 7*1 + 0.1f,
        4*1 + 5*0 + 6*(-1) + 5*0 + 6*1 + 7*0 + 6*(-1) + 7*0 + 8*1 + 0.1f,
        5*1 + 6*0 + 7*(-1) + 6*0 + 7*1 + 8*0 + 7*(-1) + 8*0 + 9*1 + 0.1f
    };

    // Check the output values
    for (size_t i = 0; i < expectedOutput.size(); ++i) {
        ASSERT_NEAR(output.getData()[i], expectedOutput[i], 1e-5);
    }
}

TEST(Conv2DLayerTest, BackwardPassHardcodedConv2D) {
    Conv2DLayer<float> convLayer(1, 1, 3); // 1 input channel, 1 output channel, 3x3 kernel

    // Set weights and biases
    Tensor<float> weightValues({1, 1, 3, 3}, {1, 0, -1,
                                              0, 1, 0,
                                              -1, 0, 1});
    Tensor<float> biasValues({1, 1}, {0.1f});

    convLayer.setWeights(weightValues);
    convLayer.setBiases(biasValues);

    // Create an input tensor (1 batch, 1 channel, 5x5 image)
    Tensor<float> input({1, 1, 5, 5}, {1, 2, 3, 4, 5,
                                       2, 3, 4, 5, 6,
                                       3, 4, 5, 6, 7,
                                       4, 5, 6, 7, 8,
                                       5, 6, 7, 8, 9});

    // Perform forward pass
    Tensor<float> output = convLayer.forward(input);

    // Create gradient output for backward pass (1 batch, 1 channel, 3x3 output)
    Tensor<float> gradOutput({1, 1, 3, 3}, {1, 1, 1,
                                            1, 1, 1,
                                            1, 1, 1});

    // Perform backward pass
    Tensor<float> gradInput = convLayer.backward(gradOutput);

    // Calculate expected input gradient manually
    std::vector<std::vector<float>> expectedGradInput(5, std::vector<float>(5, 0.0f));
    for (int oh = 0; oh < 3; ++oh) {
        for (int ow = 0; ow < 3; ++ow) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    expectedGradInput[ih][iw] += weightValues.at({0, 0, kh, kw}) * gradOutput.at({0, 0, oh, ow});
                }
            }
        }
    }

    // Check the input gradient values
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            ASSERT_NEAR(gradInput.at({0, 0, i, j}), expectedGradInput[i][j], 1e-5)
                << "Mismatch at position (" << i << ", " << j << ")";
        }
    }

    // Check weight gradients
    Tensor<float> weightGradients = convLayer.getWeightGradients();
    ASSERT_EQ(weightGradients.getShape(), Shape({1, 1, 3, 3}));

    // Calculate expected weight gradients
    std::vector<float> expectedWeightGradients = {
        27, 36, 45,
        36, 45, 54,
        45, 54, 63
    };

    for (size_t i = 0; i < expectedWeightGradients.size(); ++i) {
        ASSERT_NEAR(weightGradients.getData()[i], expectedWeightGradients[i], 1e-5)
            << "Weight gradient mismatch at index " << i;
    }

    // Check bias gradients
    Tensor<float> biasGradients = convLayer.getBiasGradients();
    ASSERT_EQ(biasGradients.getShape(), Shape({1, 1}));
    ASSERT_NEAR(biasGradients.getData()[0], 9.0f, 1e-5); // Sum of all elements in gradOutput
}

TEST(Conv2DLayerTest, DifferentInputShapes) {
    Conv2DLayer<float> convLayer(3, 2, 3); // 3 input channels, 2 output channels, 3x3 kernel

    // Test with single sample (4D input with batch size 1)
    Tensor<float> input1({1, 3, 32, 32});
    Tensor<float> output1 = convLayer.forward(input1);
    ASSERT_EQ(output1.getShape(), Shape({1, 2, 30, 30}));

    // Test with multiple samples (4D input with batch size > 1)
    Tensor<float> input2({5, 3, 32, 32});
    Tensor<float> output2 = convLayer.forward(input2);
    ASSERT_EQ(output2.getShape(), Shape({5, 2, 30, 30}));

    // Perform backward pass with multiple samples
    Tensor<float> gradOutput({5, 2, 30, 30});
    Tensor<float> gradInput = convLayer.backward(gradOutput);
    ASSERT_EQ(gradInput.getShape(), Shape({5, 3, 32, 32}));
}

TEST(Conv2DLayerTest, WeightUpdateWithOptimizer) {
    Conv2DLayer<float> convLayer(1, 1, 3);
    AdamOptions adamOptions;
    adamOptions.learningRate = 0.01f;
    AdamOptimizer<float> optimizer(adamOptions);

    // Perform a forward and backward pass
    Tensor<float> input({1, 1, 5, 5}, 1.0f);  // Initialize with 1s for simplicity
    Tensor<float> output = convLayer.forward(input);
    Tensor<float> gradOutput({1, 1, 3, 3}, 1.0f);  // Initialize with 1s for simplicity
    convLayer.backward(gradOutput);

    // Get initial weights and biases
    Tensor<float> initialWeights = convLayer.getWeights();
    Tensor<float> initialBiases = convLayer.getBiases();

    // Perform weight update
    convLayer.updateWeights(optimizer);

    // Get updated weights and biases
    Tensor<float> updatedWeights = convLayer.getWeights();
    Tensor<float> updatedBiases = convLayer.getBiases();

    // Check that weights and biases have been updated
    ASSERT_FALSE(TensorEquals(initialWeights, updatedWeights));
    ASSERT_FALSE(TensorEquals(initialBiases, updatedBiases));
}

} // namespace smart_dnn

#endif // TEST_CONV_2D_CPP
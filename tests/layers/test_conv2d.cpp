#ifndef TEST_CONV2D_CPP
#define TEST_CONV2D_CPP

#include <gtest/gtest.h>
#include "tests/utils/tensor_helpers.hpp"
#include "smart_dnn/optimizers/AdamOptimizer.hpp"
#include "smart_dnn/layers/Conv2DLayer.hpp"

namespace sdnn {


TEST(Conv2DLayerTest, ForwardPassHardcodedConv2D) {
    Conv2DLayer convLayer(1, 1, 3);  // 1 input channel, 1 output channel, 3x3 kernel

    // Set weights and biases
    Tensor weightValues({1, 1, 3, 3}, {1.0f, 0.0f, -1.0f,
                                       0.0f, 1.0f, 0.0f,
                                       -1.0f, 0.0f, 1.0f});
    Tensor biasValues({1, 1}, {0.1f});

    convLayer.setWeights(weightValues);
    convLayer.setBiases(biasValues);

    // Create an input tensor (1 batch, 1 channel, 5x5 image)
    Tensor input({1, 1, 5, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                                 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

    // Perform forward pass
    Tensor output = convLayer.forward(input);

    // Expected output shape: (1, 1, 3, 3)
    ASSERT_EQ(output.shape(), Shape({1, 1, 3, 3}));

    // Calculate expected output manually
    std::vector<float> expectedOutput = {
        1*1 + 2*0 + 3*(-1) + 2*0 + 3*1 + 4*0 + 3*(-1) + 4*0 + 5*1 + 0.1f,
        2*1 + 3*0 + 4*(-1) + 3*0 + 4*1 + 5*0 + 4*(-1) + 5*0 + 6*1 + 0.1f,
        3*1 + 4*0 + 5*(-1) + 4*0 + 5*1 + 6*0 + 5*(-1) + 6*0 + 7*1 + 0.1f
    };

    // Validate output by slicing and comparing
    for (size_t i = 0; i < expectedOutput.size(); ++i) {
        ASSERT_NEAR(output.at<float>(i), expectedOutput[i], 1e-5);
    }
}

TEST(Conv2DLayerTest, BackwardPassHardcodedConv2D) {
    Conv2DLayer convLayer(1, 1, 3);  // 1 input channel, 1 output channel, 3x3 kernel

    // Set weights and biases
    Tensor weightValues({1, 1, 3, 3}, {1.0f, 0.0f, -1.0f,
                                       0.0f, 1.0f, 0.0f,
                                       -1.0f, 0.0f, 1.0f});
    Tensor biasValues({1, 1}, {0.1f});

    convLayer.setWeights(weightValues);
    convLayer.setBiases(biasValues);

    // Create an input tensor (1 batch, 1 channel, 5x5 image)
    Tensor input({1, 1, 5, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f    ,
                                 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                                 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

    // Perform forward pass
    Tensor output = convLayer.forward(input);

    // Create gradient output for backward pass (1 batch, 1 channel, 3x3 output)
    Tensor gradOutput({1, 1, 3, 3}, 1.0f);

    // Perform backward pass
    Tensor gradInput = convLayer.backward(gradOutput);

    // Calculate expected input gradient manually
    std::vector<std::vector<float>> expectedGradInput(5, std::vector<float>(5, 0.0f));
    for (int oh = 0; oh < 3; ++oh) {
        for (int ow = 0; ow < 3; ++ow) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    expectedGradInput[ih][iw] += weightValues.at<float>({0, 0, static_cast<size_t>(kh), static_cast<size_t>(kw)}) * gradOutput.at<float>({0, 0, static_cast<size_t>(oh), static_cast<size_t>(ow)});
                }
            }
        }
    }

    // Check the input gradient values
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            ASSERT_NEAR(gradInput.at<float>({0, 0, static_cast<size_t>(i), static_cast<size_t>(j)}), expectedGradInput[i][j], 1e-5)
                << "Mismatch at position (" << i << ", " << j << ")";
        }
    }

    // Check weight gradients
    Tensor weightGradients = convLayer.getWeightGradients();
    ASSERT_EQ(weightGradients.shape(), Shape({1, 1, 3, 3}));

    // Calculate expected weight gradients
    std::vector<float> expectedWeightGradients = {
        27, 36, 45,
        36, 45, 54,
        45, 54, 63
    };

    for (size_t i = 0; i < expectedWeightGradients.size(); ++i) {
        ASSERT_NEAR(weightGradients.at<float>(i), expectedWeightGradients[i], 1e-5)
            << "Weight gradient mismatch at index " << i;
    }

    // Check bias gradients
    Tensor biasGradients = convLayer.getBiasGradients();
    ASSERT_EQ(biasGradients.shape(), Shape({1, 1}));
    ASSERT_NEAR(biasGradients.at<float>(0), 9.0f, 1e-5);
}

TEST(Conv2DLayerTest, DifferentInputShapes) {
    Conv2DLayer convLayer(3, 2, 3);

    Tensor input1({1, 3, 8, 8}, 1.0f); 
    Tensor output1 = convLayer.forward(input1);
    ASSERT_EQ(output1.shape(), Shape({1, 2, 6, 6})); 

    Tensor input2({2, 3, 8, 8}, 1.0f); 
    Tensor output2 = convLayer.forward(input2);
    ASSERT_EQ(output2.shape(), Shape({2, 2, 6, 6}));

    Tensor gradOutput({2, 2, 6, 6}, 1.0f);
    Tensor gradInput = convLayer.backward(gradOutput);
    ASSERT_EQ(gradInput.shape(), Shape({2, 3, 8, 8}));

    ASSERT_GT(std::abs(gradInput.at<float>(0)), 0.0f);
    
    Tensor weightGradients = convLayer.getWeightGradients();
    ASSERT_EQ(weightGradients.shape(), Shape({2, 3, 3, 3}));
    ASSERT_GT(std::abs(weightGradients.at<float>(0)), 0.0f);

    Tensor biasGradients = convLayer.getBiasGradients();
    ASSERT_EQ(biasGradients.shape(), Shape({2, 1}));
    ASSERT_GT(std::abs(biasGradients.at<float>(0)), 0.0f);
}

TEST(Conv2DLayerTest, WeightUpdateWithOptimizer) {
    Conv2DLayer convLayer(1, 1, 3);
    AdamOptions adamOptions;
    adamOptions.learningRate = 0.01f;
    AdamOptimizer optimizer(adamOptions);

    // Perform a forward and backward pass
    Tensor input({1, 1, 5, 5}, 1.0f);  // Initialize with 1s for simplicity
    Tensor output = convLayer.forward(input);
    Tensor gradOutput({1, 1, 3, 3}, 1.0f);  // Initialize with 1s for simplicity
    convLayer.backward(gradOutput);

    // Get initial weights and biases
    Tensor initialWeights = convLayer.getWeights();
    Tensor initialBiases = convLayer.getBiases();

    // Perform weight update
    convLayer.updateWeights(optimizer);

    // Get updated weights and biases
    Tensor updatedWeights = convLayer.getWeights();
    Tensor updatedBiases = convLayer.getBiases();

    // Check that weights and biases have been updated
    ASSERT_FALSE(TensorEquals(initialWeights, updatedWeights));
    ASSERT_FALSE(TensorEquals(initialBiases, updatedBiases));
}


}  // namespace smart_dnn

#endif  // TEST_CONV_2D_CPP

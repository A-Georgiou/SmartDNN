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

/*

TEST CONV2D LAYER WITH BATCHES

*/

TEST(Conv2DLayerTest, SimpleForwardPass) {
    Conv2DLayer convLayer(1, 1, 3);  // 1 input channel, 1 output channel, 3x3 kernel

    // Set weights and biases
    Tensor weightValues({1, 1, 3, 3}, {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    });
    Tensor biasValues({1, 1}, {0.1f});

    convLayer.setWeights(weightValues);
    convLayer.setBiases(biasValues);

    // Create a simple input tensor
    Tensor input({1, 1, 3, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });

    // Perform forward pass
    Tensor output = convLayer.forward(input);

    // Expected output: sum of all input values (45) + bias (0.1)
    float expectedValue = 45.1f;

    ASSERT_EQ(output.shape(), Shape({1, 1, 1, 1}));
    EXPECT_NEAR(output.at<float>({0, 0, 0, 0}), expectedValue, 1e-5);
}

TEST(Conv2DLayerTest, ForwardPassMultipleBatches) {
    Conv2DLayer convLayer(2, 3, 3);  // 2 input channels, 3 output channels, 3x3 kernel

    // Set weights and biases
    Tensor weightValues({3, 2, 3, 3}, 1.0f);  // Initialize with 1s for simplicity
    Tensor biasValues({3, 1}, 0.1f);

    convLayer.setWeights(weightValues);
    convLayer.setBiases(biasValues);

    // Create an input tensor (2 batches, 2 channels, 5x5 image)
    Tensor input({2, 2, 5, 5}, {
        // Batch 1, Channel 1
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        // Batch 1, Channel 2
        0.5f, 1.0f, 1.5f, 2.0f, 2.5f,
        1.0f, 1.5f, 2.0f, 2.5f, 3.0f,
        1.5f, 2.0f, 2.5f, 3.0f, 3.5f,
        2.0f, 2.5f, 3.0f, 3.5f, 4.0f,
        2.5f, 3.0f, 3.5f, 4.0f, 4.5f,
        // Batch 2, Channel 1
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
        0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
        0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
        0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.5f, 0.6f, 0.7f, 0.8f, 0.9f,
        // Batch 2, Channel 2
        1.0f, 1.1f, 1.2f, 1.3f, 1.4f,
        1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
        1.2f, 1.3f, 1.4f, 1.5f, 1.6f,
        1.3f, 1.4f, 1.5f, 1.6f, 1.7f,
        1.4f, 1.5f, 1.6f, 1.7f, 1.8f
    });

    // Perform forward pass
    Tensor output = convLayer.forward(input);
    
    ASSERT_EQ(output.shape(), Shape({2, 3, 3, 3}));

    // Manually calculate expected output for a single point
    float expectedValue1 = 27.0f + 13.5f + 0.1f;  // Sum for batch 1, first output
    float expectedValue2 = 36.0f + 18.0f + 0.1f;   // Sum for batch 2, first output

    EXPECT_NEAR(output.at<float>({0, 0, 0, 0}), expectedValue1, 1e-5);
    EXPECT_NEAR(output.at<float>({0, 0, 0, 1}), expectedValue2, 1e-5);
}

TEST(Conv2DLayerTest, BackwardPassMultipleBatches) {
    Conv2DLayer convLayer(2, 3, 3);  // 2 input channels, 3 output channels, 3x3 kernel

    // Set weights and biases
    Tensor weightValues({3, 2, 3, 3}, 1.0f);  // Initialize with 1s for simplicity
    Tensor biasValues({3, 1}, 0.1f);

    convLayer.setWeights(weightValues);
    convLayer.setBiases(biasValues);

    // Create an input tensor (2 batches, 2 channels, 5x5 image)
    Tensor input({2, 2, 5, 5}, 1.0f);  // Initialize with 1s for simplicity

    // Perform forward pass
    Tensor output = convLayer.forward(input);

    // Create gradient output for backward pass (2 batches, 3 channels, 3x3 output)
    Tensor gradOutput({2, 3, 3, 3}, {
        // Batch 1
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
        // Batch 2
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f
    });

    // Perform backward pass
    Tensor gradInput = convLayer.backward(gradOutput);

    // Check input gradient shape
    ASSERT_EQ(gradInput.shape(), Shape({2, 2, 5, 5}));

    // Check that input gradients for different batches are different
    EXPECT_NE(gradInput.at<float>({0, 0, 0, 0}), gradInput.at<float>({1, 0, 0, 0}));

    // Check weight gradients
    Tensor weightGradients = convLayer.getWeightGradients();
    ASSERT_EQ(weightGradients.shape(), Shape({3, 2, 3, 3}));

    // Check that weight gradients are non-zero and different for each output channel
    EXPECT_NE(weightGradients.at<float>({0, 0, 0, 0}), 0.0f);
    EXPECT_NE(weightGradients.at<float>({1, 0, 0, 0}), weightGradients.at<float>({0, 0, 0, 0}));

    // Check bias gradients
    Tensor biasGradients = convLayer.getBiasGradients();
    ASSERT_EQ(biasGradients.shape(), Shape({3, 1}));

    // Check that bias gradients are non-zero and different for each output channel
    EXPECT_NE(biasGradients.at<float>({0, 0}), 0.0f);
    EXPECT_NE(biasGradients.at<float>({1, 0}), biasGradients.at<float>({0, 0}));
}

TEST(Conv2DLayerTest, EndToEndMultipleBatches) {
    Conv2DLayer convLayer(2, 3, 3);  // 2 input channels, 3 output channels, 3x3 kernel
    AdamOptions adamOptions;
    adamOptions.learningRate = 0.01f;
    AdamOptimizer optimizer(adamOptions);

    // Create an input tensor (2 batches, 2 channels, 5x5 image)
    Tensor input({2, 2, 5, 5}, 1.0f);  // Initialize with 1s for simplicity

    // Perform forward pass
    Tensor output = convLayer.forward(input);

    // Create target output
    Tensor target({2, 3, 3, 3}, 2.0f);  // Initialize with 2s for simplicity

    // Calculate loss (MSE for simplicity)
    Tensor diff = output - target;
    Tensor loss = sum(diff * diff) / static_cast<float>(diff.shape().size());

    // Create gradient output based on MSE loss
    Tensor gradOutput = 2.0f * diff / static_cast<float>(diff.shape().size());

    // Perform backward pass
    Tensor gradInput = convLayer.backward(gradOutput);

    // Store initial weights and biases
    Tensor initialWeights = convLayer.getWeights();
    Tensor initialBiases = convLayer.getBiases();

    // Update weights
    convLayer.updateWeights(optimizer);

    // Get updated weights and biases
    Tensor updatedWeights = convLayer.getWeights();
    Tensor updatedBiases = convLayer.getBiases();

    // Check that weights and biases have been updated
    EXPECT_NE(initialWeights.at<float>({0, 0, 0, 0}), updatedWeights.at<float>({0, 0, 0, 0}));
    EXPECT_NE(initialBiases.at<float>({0, 0}), updatedBiases.at<float>({0, 0}));

    // Perform another forward pass with updated weights
    Tensor newOutput = convLayer.forward(input);

    // Calculate new loss
    Tensor newDiff = newOutput - target;
    Tensor newLoss = sum(newDiff * newDiff) / static_cast<float>(newDiff.shape().size());

    // Check that the loss has decreased
    EXPECT_LT(newLoss.at<float>(0), loss.at<float>(0));
}

TEST(Conv2DLayerTest, MultiChannelMultiBatchForwardPass) {
    Conv2DLayer convLayer(2, 2, 2);  // 2 input channels, 2 output channels, 2x2 kernel

    // Set weights
    Tensor weightValues({2, 2, 2, 2}, {
        // Output channel 1
        1.0f, 1.0f, 1.0f, 1.0f,  // Input channel 1
        1.0f, 1.0f, 1.0f, 1.0f,  // Input channel 2
        // Output channel 2
        0.5f, 0.5f, 0.5f, 0.5f,  // Input channel 1
        0.5f, 0.5f, 0.5f, 0.5f   // Input channel 2
    });
    Tensor biasValues({2, 1}, {0.1f, 0.2f});

    convLayer.setWeights(weightValues);
    convLayer.setBiases(biasValues);

    // Create an input tensor (2 batches, 2 channels, 3x3 image)
    Tensor input({2, 2, 3, 3}, {
        // Batch 1
        1.0f, 2.0f, 3.0f,  4.0f, 5.0f, 6.0f,  7.0f, 8.0f, 9.0f,   // Channel 1
        9.0f, 8.0f, 7.0f,  6.0f, 5.0f, 4.0f,  3.0f, 2.0f, 1.0f,   // Channel 2
        // Batch 2
        1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f,   // Channel 1
        2.0f, 2.0f, 2.0f,  2.0f, 2.0f, 2.0f,  2.0f, 2.0f, 2.0f    // Channel 2
    });

    // Perform forward pass
    Tensor output = convLayer.forward(input);

    // Expected output shape: (2, 2, 2, 2)
    ASSERT_EQ(output.shape(), Shape({2, 2, 2, 2}));

    // Manually calculate expected output for first batch, first output channel
    float expectedValue1 = (1+2+4+5) + (9+8+6+5) + 0.1f;  // Sum of relevant input values plus bias
    // Manually calculate expected output for second batch, second output channel
    float expectedValue2 = (1+1+1+1) * 0.5f + (2+2+2+2) * 0.5f + 0.2f;  // Sum of relevant input values times weights plus bias

    EXPECT_NEAR(output.at<float>({0, 0, 0, 0}), expectedValue1, 1e-5);
    EXPECT_NEAR(output.at<float>({1, 1, 0, 0}), expectedValue2, 1e-5);
}


}  // namespace smart_dnn

#endif  // TEST_CONV_2D_CPP

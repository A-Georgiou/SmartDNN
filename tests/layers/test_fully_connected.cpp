#ifndef TESTS_LAYERS_TEST_FULLY_CONNECTED_CPP
#define TESTS_LAYERS_TEST_FULLY_CONNECTED_CPP

#include <gtest/gtest.h>
#include "tests/utils/tensor_helpers.hpp"
#include "smart_dnn/optimizers/AdamOptimizer.hpp"
#include "smart_dnn/layers/FullyConnectedLayer.hpp"
#include "smart_dnn/regularisation/BatchNormalizationLayer.hpp"
#include "smart_dnn/regularisation/DropoutLayer.hpp"

namespace sdnn {

TEST(FullyConnectedLayerTest, ForwardPassHardcodedFullyConnected) {
    FullyConnectedLayer<float> fcLayer(2, 3);  // Input size = 2, Output size = 3

    Tensor<float> weightValues({2, 3}, {1.0f, 2.0f, 3.0f,
                                        4.0f, 5.0f, 6.0f});
    Tensor<float> biasValues({1, 3}, {0.1f, 0.2f, 0.3f});

    fcLayer.setWeights(weightValues);
    fcLayer.setBiases(biasValues);

    // Create an input tensor of shape (1, 2) with hardcoded values
    Tensor<float> input({1, 2}, {1.0f, 1.0f});  // Input: [1.0, 1.0]

    // Perform forward pass
    Tensor<float> output = fcLayer.forward(input);

    // Expected output: (input * weights) + bias
    // [1.0, 1.0] * [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] + [0.1, 0.2, 0.3]
    // => [1*1 + 1*4 + 0.1, 1*2 + 1*5 + 0.2, 1*3 + 1*6 + 0.3]
    // => [5.1, 7.2, 9.3]
    std::vector<float> expectedOutput = {5.1f, 7.2f, 9.3f};

    // Check the output values
    for (size_t i = 0; i < expectedOutput.size(); ++i) {
        ASSERT_NEAR(output.getData()[i], expectedOutput[i], 1e-5);
    }
}

TEST(FullyConnectedLayerTest, BackwardPassHardcodedFullyConnected) {
    FullyConnectedLayer<float> fcLayer(2, 3);  // Input size = 2, Output size = 3

    Tensor<float> weightValues({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> biasValues({1, 3}, {0.1f, 0.2f, 0.3f});

    fcLayer.setWeights(weightValues);
    fcLayer.setBiases(biasValues);

    // Create an input tensor (batchSize=2, inputSize=2)
    Tensor<float> input({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});  // Input: [[1, 2], [3, 4]]
    Tensor<float> gradOutput({2, 3}, {1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f});  // Gradient output: all ones

    // Perform forward pass
    fcLayer.forward(input);

    // Perform backward pass
    Tensor<float> gradInput = fcLayer.backward(gradOutput);

    // Expected input gradient = gradOutput * weights.T
    // Weights: [[1, 2, 3], [4, 5, 6]]
    // Grad output: [[1, 1, 1], [1, 1, 1]]
    // Grad input: [[1*1 + 1*2 + 1*3, 1*4 + 1*5 + 1*6], [1*1 + 1*2 + 1*3, 1*4 + 1*5 + 1*6]]
    std::vector<float> expectedGradInput = {6.0f, 15.0f, 6.0f, 15.0f};

    for (size_t i = 0; i < expectedGradInput.size(); ++i) {
        ASSERT_NEAR(gradInput.getData()[i], expectedGradInput[i], 1e-5);
    }

    // Check the weight gradients
    Tensor<float> weightGradients = fcLayer.getWeightGradients();
    // Expected weight gradients: input.T * gradOutput
    // Input: [[1, 2], [3, 4]]
    // Grad output: [[1, 1, 1], [1, 1, 1]]
    // Weight gradients: [[1*1 + 3*1, 1*1 + 3*1, 1*1 + 3*1],
    //                    [2*1 + 4*1, 2*1 + 4*1, 2*1 + 4*1]]
    std::vector<float> expectedWeightGradients = {4.0f, 4.0f, 4.0f,
                                                  6.0f, 6.0f, 6.0f};

    for (size_t i = 0; i < expectedWeightGradients.size(); ++i) {
        ASSERT_NEAR(weightGradients.getData()[i], expectedWeightGradients[i], 1e-5);
    }

    // Check the bias gradients
    Tensor<float> biasGradients = fcLayer.getBiasGradients();
    // Bias gradients are the sum of gradOutput across the batch dimension
    std::vector<float> expectedBiasGradients = {2.0f, 2.0f, 2.0f};

    for (size_t i = 0; i < expectedBiasGradients.size(); ++i) {
        ASSERT_NEAR(biasGradients.getData()[i], expectedBiasGradients[i], 1e-5);
    }
}

TEST(FullyConnectedLayerTest, ComprehensiveForwardAndBackwardPass) {
    FullyConnectedLayer<float> fcLayer(2, 3);  // Input size = 2, Output size = 3

    Tensor<float> weightValues({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> biasValues({1, 3}, {0.1f, 0.2f, 0.3f});

    fcLayer.setWeights(weightValues);
    fcLayer.setBiases(biasValues);

    // Create an input tensor (batchSize=2, inputSize=2)
    Tensor<float> input({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});  // Input: [[1, 2], [3, 4]]

    // Perform forward pass
    Tensor<float> output = fcLayer.forward(input);

    // Check forward pass results
    ASSERT_EQ(output.getShape(), Shape({2, 3}));
    std::vector<float> expectedOutput = {
        1*1 + 2*4 + 0.1f, 1*2 + 2*5 + 0.2f, 1*3 + 2*6 + 0.3f,
        3*1 + 4*4 + 0.1f, 3*2 + 4*5 + 0.2f, 3*3 + 4*6 + 0.3f
    };
    for (size_t i = 0; i < expectedOutput.size(); ++i) {
        ASSERT_NEAR(output.getData()[i], expectedOutput[i], 1e-5) << "Forward pass output mismatch at index " << i;
    }

    // Create gradient output for backward pass
    Tensor<float> gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});  // Gradient output: all ones

    // Perform backward pass
    Tensor<float> gradInput = fcLayer.backward(gradOutput);

    // Check input gradient
    ASSERT_EQ(gradInput.getShape(), Shape({2, 2}));
    std::vector<float> expectedGradInput = {6.0f, 15.0f, 6.0f, 15.0f};
    for (size_t i = 0; i < expectedGradInput.size(); ++i) {
        ASSERT_NEAR(gradInput.getData()[i], expectedGradInput[i], 1e-5) << "Input gradient mismatch at index " << i;
    }

    // Check weight gradients
    Tensor<float> weightGradients = fcLayer.getWeightGradients();
    ASSERT_EQ(weightGradients.getShape(), Shape({2, 3}));
    std::vector<float> expectedWeightGradients = {4.0f, 4.0f, 4.0f, 6.0f, 6.0f, 6.0f};
    for (size_t i = 0; i < expectedWeightGradients.size(); ++i) {
        ASSERT_NEAR(weightGradients.getData()[i], expectedWeightGradients[i], 1e-5) << "Weight gradient mismatch at index " << i;
    }

    // Check bias gradients
    Tensor<float> biasGradients = fcLayer.getBiasGradients();
    ASSERT_EQ(biasGradients.getShape(), Shape({1, 3}));
    std::vector<float> expectedBiasGradients = {2.0f, 2.0f, 2.0f};
    for (size_t i = 0; i < expectedBiasGradients.size(); ++i) {
        ASSERT_NEAR(biasGradients.getData()[i], expectedBiasGradients[i], 1e-5) << "Bias gradient mismatch at index " << i;
    }
}

TEST(FullyConnectedLayerTest, DifferentInputShapes) {
    FullyConnectedLayer<float> fcLayer(3, 2);  // Input size = 3, Output size = 2

    Tensor<float> weightValues({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> biasValues({1, 2}, {0.1f, 0.2f});

    fcLayer.setWeights(weightValues);
    fcLayer.setBiases(biasValues);

    // Test with a single sample (1D input)
    Tensor<float> input1D({3}, {1.0f, 2.0f, 3.0f});
    Tensor<float> output1D = fcLayer.forward(input1D);
    ASSERT_EQ(output1D.getShape(), Shape({2}));

    // Test with multiple samples (2D input)
    Tensor<float> input2D({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> output2D = fcLayer.forward(input2D);
    ASSERT_EQ(output2D.getShape(), Shape({2, 2}));

    // Perform backward pass with 2D input
    Tensor<float> gradOutput({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor<float> gradInput = fcLayer.backward(gradOutput);
    ASSERT_EQ(gradInput.getShape(), Shape({2, 3}));

    // Check bias gradients shape
    Tensor<float> biasGradients = fcLayer.getBiasGradients();
    ASSERT_EQ(biasGradients.getShape(), Shape({1, 2}));
}

TEST(FullyConnectedLayerTest, BatchedDataProcessing) {
    FullyConnectedLayer<float> fcLayer(2, 3);  // Input size = 2, Output size = 3

    // Set weights and biases
    Tensor<float> weightValues({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> biasValues({1, 3}, {0.1f, 0.2f, 0.3f});
    fcLayer.setWeights(weightValues);
    fcLayer.setBiases(biasValues);

    // Create batched input (batch_size = 2, input_size = 2)
    Tensor<float> batchedInput({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    // Forward pass
    Tensor<float> output = fcLayer.forward(batchedInput);

    // Check output shape
    ASSERT_EQ(output.getShape(), Shape({2, 3}));

    // Expected output calculation
    std::vector<float> expectedOutput = {
        1*1 + 2*4 + 0.1f, 1*2 + 2*5 + 0.2f, 1*3 + 2*6 + 0.3f,  // First sample
        3*1 + 4*4 + 0.1f, 3*2 + 4*5 + 0.2f, 3*3 + 4*6 + 0.3f   // Second sample
    };

    // Check forward pass results
    for (size_t i = 0; i < expectedOutput.size(); ++i) {
        ASSERT_NEAR(output.getData()[i], expectedOutput[i], 1e-5) 
            << "Forward pass output mismatch at index " << i;
    }

    // Backward pass
    Tensor<float> gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor<float> gradInput = fcLayer.backward(gradOutput);

    // Check gradInput shape
    ASSERT_EQ(gradInput.getShape(), Shape({2, 2}));

    // Expected gradInput
    std::vector<float> expectedGradInput = {6.0f, 15.0f, 6.0f, 15.0f};
    for (size_t i = 0; i < expectedGradInput.size(); ++i) {
        ASSERT_NEAR(gradInput.getData()[i], expectedGradInput[i], 1e-5) 
            << "GradInput mismatch at index " << i;
    }

    // Check weight gradients
    Tensor<float> weightGradients = fcLayer.getWeightGradients();
    ASSERT_EQ(weightGradients.getShape(), Shape({2, 3}));
    std::vector<float> expectedWeightGradients = {4.0f, 4.0f, 4.0f, 6.0f, 6.0f, 6.0f};
    for (size_t i = 0; i < expectedWeightGradients.size(); ++i) {
        ASSERT_NEAR(weightGradients.getData()[i], expectedWeightGradients[i], 1e-5) 
            << "Weight gradient mismatch at index " << i;
    }

    // Check bias gradients
    Tensor<float> biasGradients = fcLayer.getBiasGradients();
    ASSERT_EQ(biasGradients.getShape(), Shape({1, 3}));
    std::vector<float> expectedBiasGradients = {2.0f, 2.0f, 2.0f};  // Sum across batch
    for (size_t i = 0; i < expectedBiasGradients.size(); ++i) {
        ASSERT_NEAR(biasGradients.getData()[i], expectedBiasGradients[i], 1e-5) 
            << "Bias gradient mismatch at index " << i;
    }
}

} // namespace sdnn

#endif // TESTS_LAYERS_TEST_FULLY_CONNECTED_CPP
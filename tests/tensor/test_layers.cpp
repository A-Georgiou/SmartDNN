#include <gtest/gtest.h>
#include "../smart_dnn/Layers/Conv2DLayer.hpp"
#include "../utils/tensor_helpers.hpp"
#include "../smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "../smart_dnn/Regularisation/DropoutLayer.hpp"

namespace smart_dnn {

/*

    CONV2D LAYER TESTS

*/

TEST(Conv2DLayerTest, ForwardPassShape) {
    // Create a Conv2D layer with inputChannels = 1, outputChannels = 2, kernel size 3x3
    Conv2DLayer<float> convLayer(1, 2, 3);

    // Create an input tensor with shape (batchSize=4, inputChannels=1, height=5, width=5)
    Tensor<float> input({4, 1, 5, 5}, 1.0f); // Simple input tensor filled with ones

    // Perform forward pass
    Tensor<float> output = convLayer.forward(input);

    // Check that the output shape is correct
    // Output should have shape (4, 2, 3, 3) with (height and width reduced due to convolution)
    ValidateTensorShape(output, 4, 72, {4, 2, 3, 3});

    // Ensure that the output values are not NaN or infinite
    for (int i = 0; i < output.getData().size(); ++i) {
        ASSERT_TRUE(std::isfinite(output.getData()[i]));
    }
}

TEST(Conv2DLayerTest, BackwardPassShape) {
    // Create a Conv2D layer with inputChannels = 4, outputChannels = 2, kernel size 3x3
    // Stride = 1, padding = 0, dilation = 1
    Conv2DLayer<float> convLayer(1, 2, 3);

    // Create a simple input and perform forward pass
    Tensor<float> input({4, 1, 5, 5}, 1.0f);
    Tensor<float> output = convLayer.forward(input);

    // Create a gradient tensor for the output with the same shape as output
    Tensor<float> gradOutput(output.getShape(), 1.0f); // Assume simple gradient of all ones

    // Perform backward pass
    Tensor<float> gradInput = convLayer.backward(gradOutput);

    // Check that the gradient with respect to input has the correct shape
    ValidateTensorShape(gradInput, 4, 100, {4, 1, 5, 5});
}

TEST(Conv2DLayerTest, WeightInitialization) {
    Conv2DLayer<float> convLayer(1, 2, 3);

    // Retrieve weights and biases after initialization
    Tensor<float> weights = convLayer.getWeights();
    Tensor<float> biases = convLayer.getBiases();

    // Check that the weights are initialized using He initialization (with stddev based on input size)
    // You could set a range based on the expected He stddev
    float stddev = std::sqrt(2.0f / (1 * 3 * 3));
    for (int i = 0; i < weights.getData().size(); ++i) {
        ASSERT_GE(weights.getData()[i], -3 * stddev);
        ASSERT_LE(weights.getData()[i], 3 * stddev);
    }

    // Check biases initialized to 0.01
    for (int i = 0; i < biases.getData().size(); ++i) {
        ASSERT_NEAR(biases.getData()[i], 0.01f, 1e-5);
    }
}

TEST(Conv2DLayerTest, WeightUpdate) {
    Conv2DLayer<float> convLayer(1, 2, 3);

    // Create simple input and perform forward and backward passes
    Tensor<float> input({4, 1, 5, 5}, 1.0f);
    Tensor<float> output = convLayer.forward(input);

    Tensor<float> gradOutput(output.getShape(), 1.0f); // Simple gradient
    convLayer.backward(gradOutput);

    // Create an Adam optimizer
    AdamOptions<float> adamOptions;
    adamOptions.learningRate = 1e-3f;
    AdamOptimizer<float> optimizer(adamOptions);

    // Save initial weights for comparison
    Tensor<float> initialWeights = convLayer.getWeights();
    Tensor<float> initialBiases = convLayer.getBiases();

    // Perform weight update
    convLayer.updateWeights(optimizer);

    // Check that the weights and biases have been updated (not equal to initial values)
    Tensor<float> updatedWeights = convLayer.getWeights();
    Tensor<float> updatedBiases = convLayer.getBiases();

    for (int i = 0; i < updatedWeights.getData().size(); ++i) {
        ASSERT_NE(updatedWeights.getData()[i], initialWeights.getData()[i]);
    }

    for (int i = 0; i < updatedBiases.getData().size(); ++i) {
        ASSERT_NE(updatedBiases.getData()[i], initialBiases.getData()[i]);
    }
}


/*

    FULLY CONNECTED LAYER TESTS

*/

TEST(FullyConnectedLayerTest, ForwardPass) {
    FullyConnectedLayer<float> fcLayer(4, 3);  // Input size = 4, Output size = 3

    // Create an input tensor of shape (1, 4)
    Tensor<float> input({1, 4}, 1.0f);  // All values set to 1.0

    // Perform forward pass
    Tensor<float> output = fcLayer.forward(input);

    // Check the output shape (should be (1, 3) since we have 3 output neurons)
    ValidateTensorShape(output, 2, 3, {1, 3});

    // Ensure output is finite and not NaN or infinity
    for (size_t i = 0; i < output.getData().size(); ++i) {
        ASSERT_TRUE(std::isfinite(output.getData()[i]));
    }
}

TEST(FullyConnectedLayerTest, BackwardPass) {
    FullyConnectedLayer<float> fcLayer(4, 3);  // Input size = 4, Output size = 3

    // Create an input tensor (batchSize=2, inputSize=4)
    Tensor<float> input({2, 4}, 1.0f);  // All values set to 1.0
    Tensor<float> gradOutput({2, 3}, 1.0f);  // Gradient output from next layer

    // Perform forward pass
    fcLayer.forward(input);

    // Perform backward pass
    Tensor<float> gradInput = fcLayer.backward(gradOutput);

    // Check if the gradients are the correct shape
    ValidateTensorShape(gradInput, 2, 8, {2, 4});  // Gradients should have the same shape as input

    // Check if the weight gradients and bias gradients are calculated
    Tensor<float> weightGradients = fcLayer.getWeightGradients();
    Tensor<float> biasGradients = fcLayer.getBiasGradients();

    ValidateTensorShape(weightGradients, 2, 12, {4, 3});  // Weight gradients should have the shape of (inputSize, outputSize)
    
    // Adjusted to expect a 1D tensor of size 3 for biases
    ValidateTensorShape(biasGradients, 1, 3, {3});  // Bias gradients should have the shape of (outputSize)
}

TEST(FullyConnectedLayerTest, WeightUpdate) {
    FullyConnectedLayer<float> fcLayer(4, 3);  // Input size = 4, Output size = 3

    // Create an input tensor and perform forward pass
    Tensor<float> input({2, 4}, 1.0f);
    fcLayer.forward(input);

    // Create a dummy gradient tensor and perform backward pass
    Tensor<float> gradOutput({2, 3}, 1.0f);  // All gradients are 1
    fcLayer.backward(gradOutput);

    // Create an Adam optimizer
    AdamOptions<float> adamOptions;
    adamOptions.learningRate = 1e-3f;
    AdamOptimizer<float> optimizer(adamOptions);

    // Get initial weights and biases for comparison
    Tensor<float> initialWeights = fcLayer.getWeights();
    Tensor<float> initialBiases = fcLayer.getBiases();

    // Perform weight update
    fcLayer.updateWeights(optimizer);

    // Check that weights and biases have been updated
    Tensor<float> updatedWeights = fcLayer.getWeights();
    Tensor<float> updatedBiases = fcLayer.getBiases();

    for (size_t i = 0; i < updatedWeights.getData().size(); ++i) {
        ASSERT_NE(updatedWeights.getData()[i], initialWeights.getData()[i]);
    }

    for (size_t i = 0; i < updatedBiases.getData().size(); ++i) {
        ASSERT_NE(updatedBiases.getData()[i], initialBiases.getData()[i]);
    }
}

/*

    DROPOUT LAYER TESTS

*/


TEST(DropoutLayerTest, ForwardPassTraining) {
    DropoutLayer<float> dropoutLayer(0.5);  // 50% dropout

    // Input tensor: batchSize=2, features=4 (2D shape)
    Tensor<float> input({2, 4}, 1.0f);

    // Set training mode and perform forward pass
    dropoutLayer.setTrainingMode(true);
    Tensor<float> output = dropoutLayer.forward(input);

    // Check that the output shape is correct
    ValidateTensorShape(output, 2, 8, {2, 4});
    
    // The output should have 0s in approximately 50% of the elements
    float zeroCount = 0;
    for (int i = 0; i < output.getData().size(); ++i) {
        if (output.getData().data()[i] == 0) zeroCount++;
    }

    ASSERT_GT(zeroCount, 2);  // Ensure some elements are dropped
}

TEST(DropoutLayerTest, ForwardPassInference) {
    DropoutLayer<float> dropoutLayer(0.5);  // 50% dropout

    // Input tensor: batchSize=2, features=4 (2D shape)
    Tensor<float> input({2, 4}, 1.0f);

    // Set inference mode (no dropout should be applied)
    dropoutLayer.setTrainingMode(false);
    Tensor<float> output = dropoutLayer.forward(input);

    // Check that the output shape is correct
    ValidateTensorShape(output, 2, 8, {2, 4});

    // In inference mode, the output should be the same as input
    ASSERT_TRUE(input == output);  // Use tensor equality check
}

TEST(DropoutLayerTest, BackwardPass) {
    DropoutLayer<float> dropoutLayer(0.5);  // 50% dropout

    // Input tensor: batchSize=2, features=4 (2D shape)
    Tensor<float> input({2, 4}, 1.0f);

    // Set training mode and perform forward pass
    dropoutLayer.setTrainingMode(true);
    Tensor<float> output = dropoutLayer.forward(input);

    // Simulate gradient output (same shape as input)
    Tensor<float> gradOutput({2, 4}, 1.0f);

    // Perform backward pass
    Tensor<float> gradInput = dropoutLayer.backward(gradOutput);

    // Check gradient input shape
    ValidateTensorShape(gradInput, 2, 8, {2, 4});
}

TEST(FullyConnectedLayerTest, ForwardPassHardcoded) {
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

TEST(FullyConnectedLayerTest, BackwardPassHardcoded) {
    FullyConnectedLayer<float> fcLayer(2, 3);  // Input size = 2, Output size = 3

    Tensor<float> weightValues({2, 3}, {1.0f, 2.0f, 3.0f,
                                        4.0f, 5.0f, 6.0f});
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
    // Grad input: [sum([1,1,1]*[1,4]), sum([1,1,1]*[2,5]), ...]
    // For each sample:
    // First row: [1*(1+4+6), 1*(2+5+6)] => [11, 13]
    // Second row: Same => [9, 18]
    std::vector<float> expectedGradInput = {9.0f, 18.0f, 9.0f, 18.0f};

    for (size_t i = 0; i < expectedGradInput.size(); ++i) {
        ASSERT_NEAR(gradInput.getData()[i], expectedGradInput[i], 1e-5);
    }

    // Check the weight gradients
    Tensor<float> weightGradients = fcLayer.getWeightGradients();
    // Expected weight gradients: input.T * gradOutput
    // [1, 2], [3, 4] -> [1 + 3, 2 + 4] -> [[4, 4, 4], [6, 6, 6]]
    std::vector<float> expectedWeightGradients = {4.0f, 4.0f, 4.0f,
                                                  6.0f, 6.0f, 6.0f};

    for (size_t i = 0; i < expectedWeightGradients.size(); ++i) {
        ASSERT_NEAR(weightGradients.getData()[i], expectedWeightGradients[i], 1e-5);
    }

    // Check the bias gradients
    Tensor<float> biasGradients = fcLayer.getBiasGradients();
    // Bias gradients are the sum of gradOutput across the batch dimension: [2, 2, 2]
    std::vector<float> expectedBiasGradients = {2.0f, 2.0f, 2.0f};

    for (size_t i = 0; i < expectedBiasGradients.size(); ++i) {
        ASSERT_NEAR(biasGradients.getData()[i], expectedBiasGradients[i], 1e-5);
    }
}


/*

    BATCH NORM TESTS

*/

TEST(BatchNormalizationLayerTest, ForwardPass2D) {
    BatchNormalizationLayer<float> bnLayer(4);

    // Input tensor: batchSize=2, features=4 (2D shape)
    Tensor<float> input({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 
                                 5.0f, 6.0f, 7.0f, 8.0f});

    // Perform forward pass
    Tensor<float> output = bnLayer.forward(input);

    // Check output shape
    ValidateTensorShape(output, 2, 8, {2, 4});
}

TEST(BatchNormalizationLayerTest, ForwardPass4D) {
    BatchNormalizationLayer<float> bnLayer(3);

    // Input tensor: batchSize=2, channels=3, height=4, width=4 (4D shape)
    Tensor<float> input({2, 3, 4, 4}, 1.0f);  // All ones

    // Perform forward pass
    Tensor<float> output = bnLayer.forward(input);

    // Check output shape
    ValidateTensorShape(output, 4, 96, {2, 3, 4, 4});
}

TEST(BatchNormalizationLayerTest, BackwardPass) {
    BatchNormalizationLayer<float> bnLayer(4);

    // Input tensor: batchSize=2, features=4 (2D shape)
    Tensor<float> input({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 
                                 5.0f, 6.0f, 7.0f, 8.0f});

    // Perform forward pass
    Tensor<float> output = bnLayer.forward(input);

    // Simulate gradient output (same shape as input)
    Tensor<float> gradOutput({2, 4}, 1.0f);

    // Perform backward pass
    Tensor<float> gradInput = bnLayer.backward(gradOutput);

    // Check gradient input shape
    ValidateTensorShape(gradInput, 2, 8, {2, 4});
}

} // namespace smart_dnn


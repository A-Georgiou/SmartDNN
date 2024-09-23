#ifndef TEST_FULLY_CONNECTED_LAYER_CPP
#define TEST_FULLY_CONNECTED_LAYER_CPP

#include <gtest/gtest.h>
#include "tests/utils/tensor_helpers.hpp"
#include "smart_dnn/optimizers/AdamOptimizer.hpp"
#include "smart_dnn/layers/FullyConnectedLayer.hpp"

namespace sdnn {

TEST(FullyConnectedLayerTest, ForwardPass) {
    FullyConnectedLayer fcLayer(4, 3);  // Input size = 4, Output size = 3

    // Create an input tensor of shape (1, 4)
    Tensor input({1, 4}, 1.0f);  // All values set to 1.0

    // Perform forward pass
    Tensor output = fcLayer.forward(input);

    // Check the output shape (should be (1, 3) since we have 3 output neurons)
    ValidateTensorShape(output, 2, 3, {1, 3});

    // Ensure output is finite and not NaN or infinity
    for (size_t i = 0; i < output.shape().size(); ++i) {
        ASSERT_TRUE(std::isfinite(output.at<float>(i)));
    }
}

TEST(FullyConnectedLayerTest, BackwardPass) {
    FullyConnectedLayer fcLayer(4, 3);  // Input size = 4, Output size = 3

    // Create an input tensor (batchSize=2, inputSize=4)
    Tensor input({2, 4}, 1.0f);  // All values set to 1.0
    Tensor gradOutput({2, 3}, 1.0f);  // Gradient output from next layer

    // Perform forward pass
    fcLayer.forward(input);

    // Perform backward pass
    Tensor gradInput = fcLayer.backward(gradOutput);

    // Check if the gradients are the correct shape
    ValidateTensorShape(gradInput, 2, 8, {2, 4});  // Gradients should have the same shape as input

    // Check if the weight gradients and bias gradients are calculated
    Tensor weightGradients = fcLayer.getWeightGradients();
    Tensor biasGradients = fcLayer.getBiasGradients();

    ValidateTensorShape(weightGradients, 2, 12, {4, 3});  // Weight gradients should have the shape of (inputSize, outputSize)
    
    ValidateTensorShape(biasGradients, 2, 3, {1, 3});  // Bias gradients should have the shape of (1, outputSize)
}

TEST(FullyConnectedLayerTest, WeightUpdate) {
    FullyConnectedLayer fcLayer(4, 3);  // Input size = 4, Output size = 3

    // Create an input tensor and perform forward pass
    Tensor input({2, 4}, 1.0f);
    fcLayer.forward(input);

    // Create a dummy gradient tensor and perform backward pass
    Tensor gradOutput({2, 3}, 1.0f);  // All gradients are 1
    fcLayer.backward(gradOutput);

    // Create an Adam optimizer
    AdamOptions adamOptions;
    adamOptions.learningRate = 1e-3f;
    AdamOptimizer optimizer(adamOptions);

    // Get initial weights and biases for comparison
    Tensor initialWeights = fcLayer.getWeights();
    Tensor initialBiases = fcLayer.getBiases();

    // Perform weight update
    fcLayer.updateWeights(optimizer);

    // Check that weights and biases have been updated
    Tensor updatedWeights = fcLayer.getWeights();
    Tensor updatedBiases = fcLayer.getBiases();

    std::cout << "Initial weights: " << initialWeights.toString() << std::endl;
    std::cout << "Updated weights: " << updatedWeights.toString() << std::endl;

    std::cout << "Initial biases: " << initialBiases.toString() << std::endl;
    std::cout << "Updated biases: " << updatedBiases.toString() << std::endl;


    for (size_t i = 0; i < updatedWeights.shape().size(); ++i) {
        ASSERT_NE(updatedWeights.at<float>(i), initialWeights.at<float>(i));
    }

    for (size_t i = 0; i < updatedBiases.shape().size(); ++i) {
        ASSERT_NE(updatedBiases.at<float>(i), initialBiases.at<float>(i));
    }
}

} // namespace sdnn 

#endif // TEST_FULLY_CONNECTED_LAYER_CPP

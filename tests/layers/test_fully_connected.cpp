
#include <gtest/gtest.h>
#include "../smart_dnn/Layers/Conv2DLayer.hpp"
#include "../utils/tensor_helpers.hpp"
#include "../smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "../smart_dnn/Regularisation/DropoutLayer.hpp"

namespace smart_dnn {

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


}
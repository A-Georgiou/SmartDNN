#ifndef TEST_ADDITIONAL_LAYERS_CPP
#define TEST_ADDITIONAL_LAYERS_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Layers/FlattenLayer.hpp"
#include "../../smart_dnn/Layers/ActivationLayer.hpp"
#include "../../smart_dnn/Regularisation/MaxPooling2DLayer.hpp"
#include "../../smart_dnn/Activations/ReLU.hpp"
#include "../../smart_dnn/Activations/Sigmoid.hpp"
#include "../utils/tensor_helpers.hpp"

namespace smart_dnn {

// ==================== Flatten Layer Tests ====================

TEST(FlattenLayerTest, Flatten3DTo2D) {
    FlattenLayer<float> flattenLayer;
    
    // Input: batch size 2, 3 channels, 4x5 spatial dimensions
    Tensor<float> input({2, 3, 4, 5}, 1.0f);
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Output should be (batch_size, flattened_size) = (2, 3*4*5) = (2, 60)
    EXPECT_EQ(output.getShape(), Shape({2, 60}));
    EXPECT_EQ(output.getShape().size(), input.getShape().size());
}

TEST(FlattenLayerTest, Flatten4DTo2D) {
    FlattenLayer<float> flattenLayer;
    
    // Input: batch size 3, 2 channels, 4x4 spatial dimensions
    Tensor<float> input({3, 2, 4, 4}, 5.0f);
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Output should be (3, 2*4*4) = (3, 32)
    EXPECT_EQ(output.getShape(), Shape({3, 32}));
    
    // Verify all values are preserved
    for (int i = 0; i < output.getShape().size(); ++i) {
        EXPECT_FLOAT_EQ(output.getData()[i], 5.0f);
    }
}

TEST(FlattenLayerTest, AlreadyFlat2D) {
    FlattenLayer<float> flattenLayer;
    
    // Input already 2D
    Tensor<float> input({5, 10}, 2.0f);
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Output should remain the same shape
    EXPECT_EQ(output.getShape(), Shape({5, 10}));
}

TEST(FlattenLayerTest, BackwardPass) {
    FlattenLayer<float> flattenLayer;
    
    // Forward pass to store original shape
    Tensor<float> input({2, 3, 4, 5}, 1.0f);
    Tensor<float> output = flattenLayer.forward(input);
    
    // Backward pass
    Tensor<float> gradOutput({2, 60}, 2.0f);
    Tensor<float> gradInput = flattenLayer.backward(gradOutput);
    
    // Gradient should be reshaped back to original input shape
    EXPECT_EQ(gradInput.getShape(), Shape({2, 3, 4, 5}));
    
    // Verify all gradient values are preserved
    for (int i = 0; i < gradInput.getShape().size(); ++i) {
        EXPECT_FLOAT_EQ(gradInput.getData()[i], 2.0f);
    }
}

TEST(FlattenLayerTest, InvalidInputRank) {
    FlattenLayer<float> flattenLayer;
    
    // Input with rank < 2 should throw
    Tensor<float> input({10}, 1.0f);
    
    EXPECT_THROW(flattenLayer.forward(input), std::invalid_argument);
}

// ==================== Activation Layer Tests ====================

TEST(ActivationLayerTest, ForwardWithReLU) {
    ReLU<float> relu;
    ActivationLayer<float> activationLayer(relu);
    
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> output = activationLayer.forward(input);
    
    EXPECT_EQ(output.getShape(), input.getShape());
    EXPECT_FLOAT_EQ(output.getData()[0], 0.0f);  // -2 -> 0
    EXPECT_FLOAT_EQ(output.getData()[1], 0.0f);  // -1 -> 0
    EXPECT_FLOAT_EQ(output.getData()[2], 0.0f);  // 0 -> 0
    EXPECT_FLOAT_EQ(output.getData()[3], 1.0f);  // 1 -> 1
    EXPECT_FLOAT_EQ(output.getData()[4], 2.0f);  // 2 -> 2
    EXPECT_FLOAT_EQ(output.getData()[5], 3.0f);  // 3 -> 3
}

TEST(ActivationLayerTest, BackwardWithReLU) {
    ReLU<float> relu;
    ActivationLayer<float> activationLayer(relu);
    
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    
    // Forward pass first
    activationLayer.forward(input);
    
    // Backward pass
    Tensor<float> gradInput = activationLayer.backward(gradOutput);
    
    EXPECT_EQ(gradInput.getShape(), input.getShape());
    EXPECT_FLOAT_EQ(gradInput.getData()[0], 0.0f);  // Negative input
    EXPECT_FLOAT_EQ(gradInput.getData()[1], 0.0f);  // Negative input
    EXPECT_FLOAT_EQ(gradInput.getData()[2], 0.0f);  // Zero input
    EXPECT_FLOAT_EQ(gradInput.getData()[3], 1.0f);  // Positive input
    EXPECT_FLOAT_EQ(gradInput.getData()[4], 1.0f);  // Positive input
    EXPECT_FLOAT_EQ(gradInput.getData()[5], 1.0f);  // Positive input
}

TEST(ActivationLayerTest, ForwardWithSigmoid) {
    Sigmoid<float> sigmoid;
    ActivationLayer<float> activationLayer(sigmoid);
    
    Tensor<float> input({1, 2}, {0.0f, 1.0f});
    Tensor<float> output = activationLayer.forward(input);
    
    EXPECT_EQ(output.getShape(), input.getShape());
    EXPECT_NEAR(output.getData()[0], 0.5f, 1e-5f);  // sigmoid(0) = 0.5
    EXPECT_NEAR(output.getData()[1], 1.0f / (1.0f + std::exp(-1.0f)), 1e-5f);  // sigmoid(1)
}

TEST(ActivationLayerTest, DifferentShapes) {
    ReLU<float> relu;
    ActivationLayer<float> activationLayer(relu);
    
    // Test with 1D tensor
    Tensor<float> input1D({5}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> output1D = activationLayer.forward(input1D);
    EXPECT_EQ(output1D.getShape(), Shape({5}));
    
    // Test with 3D tensor
    Tensor<float> input3D({2, 3, 4}, -1.0f);
    Tensor<float> output3D = activationLayer.forward(input3D);
    EXPECT_EQ(output3D.getShape(), Shape({2, 3, 4}));
    
    // All values should be 0 (ReLU of -1)
    for (int i = 0; i < output3D.getShape().size(); ++i) {
        EXPECT_FLOAT_EQ(output3D.getData()[i], 0.0f);
    }
}

// ==================== MaxPooling2D Layer Tests ====================

TEST(MaxPooling2DLayerTest, ForwardPassBasic) {
    MaxPooling2DLayer<float> poolLayer(2, 2);  // 2x2 pooling, stride 2
    
    // Input: 1 batch, 1 channel, 4x4 image
    Tensor<float> input({1, 1, 4, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });
    
    Tensor<float> output = poolLayer.forward(input);
    
    // Output should be 1x1x2x2 (halved in spatial dimensions)
    EXPECT_EQ(output.getShape(), Shape({1, 1, 2, 2}));
    
    // Check max values in each 2x2 window
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 6.0f);   // max of (1,2,5,6)
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 1}), 8.0f);   // max of (3,4,7,8)
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 0}), 14.0f);  // max of (9,10,13,14)
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 1}), 16.0f);  // max of (11,12,15,16)
}

TEST(MaxPooling2DLayerTest, ForwardPassMultipleChannels) {
    MaxPooling2DLayer<float> poolLayer(2, 2);
    
    // Input: 1 batch, 2 channels, 4x4 image
    Tensor<float> input({1, 2, 4, 4});
    
    // Fill first channel with increasing values
    for (int i = 0; i < 16; ++i) {
        input.at({0, 0, i/4, i%4}) = static_cast<float>(i + 1);
    }
    
    // Fill second channel with different values
    for (int i = 0; i < 16; ++i) {
        input.at({0, 1, i/4, i%4}) = static_cast<float>(20 - i);
    }
    
    Tensor<float> output = poolLayer.forward(input);
    
    // Output should be 1x2x2x2 (2 channels preserved)
    EXPECT_EQ(output.getShape(), Shape({1, 2, 2, 2}));
    
    // Check first channel max values
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 6.0f);
    
    // Check second channel max values
    EXPECT_FLOAT_EQ(output.at({0, 1, 0, 0}), 20.0f);
}

TEST(MaxPooling2DLayerTest, ForwardPassBatched) {
    MaxPooling2DLayer<float> poolLayer(2, 2);
    
    // Input: 3 batches, 1 channel, 4x4 image
    Tensor<float> input({3, 1, 4, 4}, 1.0f);
    
    // Set different max values for each batch
    input.at({0, 0, 0, 0}) = 10.0f;
    input.at({1, 0, 1, 1}) = 20.0f;
    input.at({2, 0, 2, 2}) = 30.0f;
    
    Tensor<float> output = poolLayer.forward(input);
    
    // Output should be 3x1x2x2
    EXPECT_EQ(output.getShape(), Shape({3, 1, 2, 2}));
    
    // Check that max values are preserved in output
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(output.at({1, 0, 0, 0}), 20.0f);
    EXPECT_FLOAT_EQ(output.at({2, 0, 1, 1}), 30.0f);
}

TEST(MaxPooling2DLayerTest, BackwardPass) {
    MaxPooling2DLayer<float> poolLayer(2, 2);
    
    // Simple 2x2 input
    Tensor<float> input({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    // Forward pass
    Tensor<float> output = poolLayer.forward(input);
    
    // Gradient from next layer
    Tensor<float> gradOutput({1, 1, 1, 1}, {10.0f});
    
    // Backward pass
    Tensor<float> gradInput = poolLayer.backward(gradOutput);
    
    // Gradient should only flow to the max element (4.0 at position [1,1])
    EXPECT_EQ(gradInput.getShape(), Shape({1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(gradInput.at({0, 0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(gradInput.at({0, 0, 0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(gradInput.at({0, 0, 1, 0}), 0.0f);
    EXPECT_FLOAT_EQ(gradInput.at({0, 0, 1, 1}), 10.0f);  // Max element gets gradient
}

TEST(MaxPooling2DLayerTest, InvalidInputRank) {
    MaxPooling2DLayer<float> poolLayer(2, 2);
    
    // Input with wrong rank should throw
    Tensor<float> input({4, 4}, 1.0f);  // 2D tensor instead of 4D
    
    EXPECT_THROW(poolLayer.forward(input), std::runtime_error);
}

TEST(MaxPooling2DLayerTest, DifferentStrides) {
    MaxPooling2DLayer<float> poolLayer(2, 1);  // 2x2 pooling, stride 1 (overlapping)
    
    // Input: 1 batch, 1 channel, 3x3 image
    Tensor<float> input({1, 1, 3, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });
    
    Tensor<float> output = poolLayer.forward(input);
    
    // With stride 1, output should be 1x1x2x2
    EXPECT_EQ(output.getShape(), Shape({1, 1, 2, 2}));
    
    // Check overlapping max values
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 5.0f);  // max of (1,2,4,5)
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 1}), 6.0f);  // max of (2,3,5,6)
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 0}), 8.0f);  // max of (4,5,7,8)
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 1}), 9.0f);  // max of (5,6,8,9)
}

} // namespace smart_dnn

#endif // TEST_ADDITIONAL_LAYERS_CPP

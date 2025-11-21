#ifndef TEST_ADDITIONAL_LAYERS_CPP
#define TEST_ADDITIONAL_LAYERS_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Layers/FlattenLayer.hpp"
#include "../../smart_dnn/Layers/ActivationLayer.hpp"
#include "../../smart_dnn/Regularisation/MaxPooling2DLayer.hpp"
#include "../../smart_dnn/Activations/ReLU.hpp"
#include "../utils/tensor_helpers.hpp"

namespace smart_dnn {

// Helper function for approximate equality
static bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// ============================================================================
// FlattenLayer Tests
// ============================================================================

TEST(FlattenLayerTest, Forward2D) {
    FlattenLayer<float> layer;
    
    // 2D tensor should remain unchanged
    Tensor<float> input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> output = layer.forward(input);
    
    EXPECT_EQ(output.getShape(), input.getShape());
    ValidateTensorData(output, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

TEST(FlattenLayerTest, Forward3D) {
    FlattenLayer<float> layer;
    
    // 3D tensor (batch_size=2, height=2, width=2)
    Tensor<float> input({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor<float> output = layer.forward(input);
    
    // Should flatten to (batch_size=2, flattened_size=4)
    EXPECT_EQ(output.getShape().rank(), 2);
    EXPECT_EQ(output.getShape()[0], 2);  // Batch size preserved
    EXPECT_EQ(output.getShape()[1], 4);  // 2*2 = 4
    ValidateTensorData(output, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
}

TEST(FlattenLayerTest, Forward4D) {
    FlattenLayer<float> layer;
    
    // 4D tensor (batch_size=2, channels=2, height=2, width=2)
    Tensor<float> input({2, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 
                                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    Tensor<float> output = layer.forward(input);
    
    // Should flatten to (batch_size=2, flattened_size=8)
    EXPECT_EQ(output.getShape().rank(), 2);
    EXPECT_EQ(output.getShape()[0], 2);  // Batch size preserved
    EXPECT_EQ(output.getShape()[1], 8);  // 2*2*2 = 8
    
    // Verify data is preserved
    std::vector<float> expectedData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 
                                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    ValidateTensorData(output, expectedData);
}

TEST(FlattenLayerTest, BackwardPass) {
    FlattenLayer<float> layer;
    
    // Forward pass to set original shape
    Tensor<float> input({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor<float> output = layer.forward(input);
    
    // Gradient with shape (2, 4)
    Tensor<float> gradOutput({2, 4}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
    Tensor<float> gradInput = layer.backward(gradOutput);
    
    // Should reshape back to original (2, 2, 2)
    EXPECT_EQ(gradInput.getShape(), input.getShape());
    ValidateTensorData(gradInput, std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
}

TEST(FlattenLayerTest, InvalidInput1D) {
    FlattenLayer<float> layer;
    
    // 1D tensor should throw
    Tensor<float> input({5}, 1.0f);
    EXPECT_THROW(layer.forward(input), std::invalid_argument);
}

// ============================================================================
// MaxPooling2DLayer Tests
// ============================================================================

TEST(MaxPooling2DLayerTest, ForwardBasic) {
    MaxPooling2DLayer<float> layer(2, 2);  // pool_size=2, stride=2
    
    // Input: batch=1, channels=1, height=4, width=4
    Tensor<float> input({1, 1, 4, 4}, {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f,  10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });
    
    Tensor<float> output = layer.forward(input);
    
    // Output should be (1, 1, 2, 2) with max values from each 2x2 pool
    EXPECT_EQ(output.getShape()[0], 1);
    EXPECT_EQ(output.getShape()[1], 1);
    EXPECT_EQ(output.getShape()[2], 2);
    EXPECT_EQ(output.getShape()[3], 2);
    
    // Expected max values:
    // Top-left: max(1, 2, 5, 6) = 6
    // Top-right: max(3, 4, 7, 8) = 8
    // Bottom-left: max(9, 10, 13, 14) = 14
    // Bottom-right: max(11, 12, 15, 16) = 16
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 1}), 8.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 0}), 14.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 1}), 16.0f);
}

TEST(MaxPooling2DLayerTest, ForwardWithMultipleChannels) {
    MaxPooling2DLayer<float> layer(2, 2);
    
    // Input: batch=1, channels=2, height=2, width=2
    Tensor<float> input({1, 2, 2, 2}, {
        // Channel 0
        1.0f, 2.0f,
        3.0f, 4.0f,
        // Channel 1
        5.0f, 6.0f,
        7.0f, 8.0f
    });
    
    Tensor<float> output = layer.forward(input);
    
    // Output should be (1, 2, 1, 1)
    EXPECT_EQ(output.getShape()[0], 1);
    EXPECT_EQ(output.getShape()[1], 2);
    EXPECT_EQ(output.getShape()[2], 1);
    EXPECT_EQ(output.getShape()[3], 1);
    
    // Channel 0 max: 4, Channel 1 max: 8
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1, 0, 0}), 8.0f);
}

TEST(MaxPooling2DLayerTest, ForwardWithStride1) {
    MaxPooling2DLayer<float> layer(2, 1);  // pool_size=2, stride=1
    
    // Input: batch=1, channels=1, height=3, width=3
    Tensor<float> input({1, 1, 3, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });
    
    Tensor<float> output = layer.forward(input);
    
    // Output should be (1, 1, 2, 2)
    EXPECT_EQ(output.getShape()[0], 1);
    EXPECT_EQ(output.getShape()[1], 1);
    EXPECT_EQ(output.getShape()[2], 2);
    EXPECT_EQ(output.getShape()[3], 2);
    
    // Expected max values with overlapping pools:
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 5.0f);  // max(1,2,4,5)
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 1}), 6.0f);  // max(2,3,5,6)
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 0}), 8.0f);  // max(4,5,7,8)
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 1}), 9.0f);  // max(5,6,8,9)
}

TEST(MaxPooling2DLayerTest, BackwardPass) {
    MaxPooling2DLayer<float> layer(2, 2);
    
    // Input: batch=1, channels=1, height=2, width=2
    Tensor<float> input({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    // Forward pass
    Tensor<float> output = layer.forward(input);
    
    // Gradient output: (1, 1, 1, 1)
    Tensor<float> gradOutput({1, 1, 1, 1}, {1.0f});
    
    // Backward pass
    Tensor<float> gradInput = layer.backward(gradOutput);
    
    // Gradient should only flow to the max element (4.0 at position [0,0,1,1])
    EXPECT_EQ(gradInput.getShape(), input.getShape());
    EXPECT_FLOAT_EQ(gradInput.at({0, 0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(gradInput.at({0, 0, 0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(gradInput.at({0, 0, 1, 0}), 0.0f);
    EXPECT_FLOAT_EQ(gradInput.at({0, 0, 1, 1}), 1.0f);
}

TEST(MaxPooling2DLayerTest, InvalidInputRank) {
    MaxPooling2DLayer<float> layer(2, 2);
    
    // 3D tensor should throw
    Tensor<float> input({2, 2, 2}, 1.0f);
    EXPECT_THROW(layer.forward(input), std::runtime_error);
}

TEST(MaxPooling2DLayerTest, BackwardWithoutForward) {
    MaxPooling2DLayer<float> layer(2, 2);
    
    // Try backward without forward
    Tensor<float> gradOutput({1, 1, 1, 1}, 1.0f);
    EXPECT_THROW(layer.backward(gradOutput), std::runtime_error);
}

// ============================================================================
// ActivationLayer Tests
// ============================================================================

TEST(ActivationLayerTest, ForwardWithReLU) {
    ReLU<float> relu;
    ActivationLayer<float> layer(relu);
    
    Tensor<float> input({2, 3}, {-1.0f, 0.0f, 1.0f, -2.0f, 3.0f, -4.0f});
    Tensor<float> output = layer.forward(input);
    
    EXPECT_EQ(output.getShape(), input.getShape());
    
    // ReLU: max(0, x)
    EXPECT_FLOAT_EQ(output.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(output.at({0, 2}), 1.0f);
    EXPECT_FLOAT_EQ(output.at({1, 0}), 0.0f);
    EXPECT_FLOAT_EQ(output.at({1, 1}), 3.0f);
    EXPECT_FLOAT_EQ(output.at({1, 2}), 0.0f);
}

TEST(ActivationLayerTest, BackwardWithReLU) {
    ReLU<float> relu;
    ActivationLayer<float> layer(relu);
    
    Tensor<float> input({2, 2}, {-1.0f, 2.0f, 0.0f, 3.0f});
    
    // Forward pass
    Tensor<float> output = layer.forward(input);
    
    // Backward pass
    Tensor<float> gradOutput({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor<float> gradInput = layer.backward(gradOutput);
    
    EXPECT_EQ(gradInput.getShape(), input.getShape());
    
    // Gradient for ReLU: 0 if input <= 0, 1 if input > 0
    EXPECT_FLOAT_EQ(gradInput.at({0, 0}), 0.0f);  // input was -1
    EXPECT_FLOAT_EQ(gradInput.at({0, 1}), 1.0f);  // input was 2
    EXPECT_FLOAT_EQ(gradInput.at({1, 0}), 0.0f);  // input was 0
    EXPECT_FLOAT_EQ(gradInput.at({1, 1}), 1.0f);  // input was 3
}

TEST(ActivationLayerTest, PreserveInputShape) {
    ReLU<float> relu;
    ActivationLayer<float> layer(relu);
    
    // Test with various shapes
    Tensor<float> input1D({5}, 1.0f);
    Tensor<float> output1D = layer.forward(input1D);
    EXPECT_EQ(output1D.getShape(), input1D.getShape());
    
    Tensor<float> input3D({2, 3, 4}, 1.0f);
    Tensor<float> output3D = layer.forward(input3D);
    EXPECT_EQ(output3D.getShape(), input3D.getShape());
}

} // namespace smart_dnn

#endif // TEST_ADDITIONAL_LAYERS_CPP

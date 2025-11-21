#ifndef TEST_POOLING_AND_FLATTEN_LAYERS_CPP
#define TEST_POOLING_AND_FLATTEN_LAYERS_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Regularisation/MaxPooling2DLayer.hpp"
#include "../../smart_dnn/Layers/FlattenLayer.hpp"
#include <cmath>

namespace smart_dnn {

// Helper function to check if two floats are approximately equal
static inline bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// ========== MaxPooling2D Tests ==========

TEST(MaxPooling2DLayerTest, ForwardPassBasic) {
    MaxPooling2DLayer<float> poolLayer(2, 2);  // 2x2 pool, stride 2
    
    // Input: 1 batch, 1 channel, 4x4 image
    Tensor<float> input({1, 1, 4, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });
    
    Tensor<float> output = poolLayer.forward(input);
    
    // Output should be 1x1x2x2 (pooling reduces size by factor of 2)
    ASSERT_EQ(output.getShape(), Shape({1, 1, 2, 2}));
    
    // Each 2x2 window's maximum
    EXPECT_TRUE(approxEqual(output.at({0, 0, 0, 0}), 6.0f));   // max(1,2,5,6)
    EXPECT_TRUE(approxEqual(output.at({0, 0, 0, 1}), 8.0f));   // max(3,4,7,8)
    EXPECT_TRUE(approxEqual(output.at({0, 0, 1, 0}), 14.0f));  // max(9,10,13,14)
    EXPECT_TRUE(approxEqual(output.at({0, 0, 1, 1}), 16.0f));  // max(11,12,15,16)
}

TEST(MaxPooling2DLayerTest, ForwardPassMultipleChannels) {
    MaxPooling2DLayer<float> poolLayer(2, 2);
    
    // Input: 1 batch, 2 channels, 4x4 image
    Tensor<float> input({1, 2, 4, 4}, {
        // Channel 0
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
        // Channel 1
        16.0f, 15.0f, 14.0f, 13.0f,
        12.0f, 11.0f, 10.0f, 9.0f,
        8.0f, 7.0f, 6.0f, 5.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    });
    
    Tensor<float> output = poolLayer.forward(input);
    
    ASSERT_EQ(output.getShape(), Shape({1, 2, 2, 2}));
    
    // Check channel 0
    EXPECT_TRUE(approxEqual(output.at({0, 0, 0, 0}), 6.0f));
    EXPECT_TRUE(approxEqual(output.at({0, 0, 1, 1}), 16.0f));
    
    // Check channel 1
    EXPECT_TRUE(approxEqual(output.at({0, 1, 0, 0}), 16.0f));
    EXPECT_TRUE(approxEqual(output.at({0, 1, 1, 1}), 6.0f));
}

TEST(MaxPooling2DLayerTest, ForwardPassBatch) {
    MaxPooling2DLayer<float> poolLayer(2, 2);
    
    // Input: 2 batches, 1 channel, 2x2 image
    Tensor<float> input({2, 1, 2, 2}, {
        // Batch 0
        1.0f, 2.0f,
        3.0f, 4.0f,
        // Batch 1
        5.0f, 6.0f,
        7.0f, 8.0f
    });
    
    Tensor<float> output = poolLayer.forward(input);
    
    ASSERT_EQ(output.getShape(), Shape({2, 1, 1, 1}));
    
    EXPECT_TRUE(approxEqual(output.at({0, 0, 0, 0}), 4.0f));
    EXPECT_TRUE(approxEqual(output.at({1, 0, 0, 0}), 8.0f));
}

TEST(MaxPooling2DLayerTest, BackwardPassBasic) {
    MaxPooling2DLayer<float> poolLayer(2, 2);
    
    // Input: 1 batch, 1 channel, 4x4 image
    Tensor<float> input({1, 1, 4, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });
    
    poolLayer.forward(input);
    
    // Gradient from next layer
    Tensor<float> gradOutput({1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    
    Tensor<float> gradInput = poolLayer.backward(gradOutput);
    
    ASSERT_EQ(gradInput.getShape(), input.getShape());
    
    // Gradient should only flow through max positions
    EXPECT_TRUE(approxEqual(gradInput.at({0, 0, 1, 1}), 1.0f));  // Position of 6
    EXPECT_TRUE(approxEqual(gradInput.at({0, 0, 1, 3}), 1.0f));  // Position of 8
    EXPECT_TRUE(approxEqual(gradInput.at({0, 0, 3, 1}), 1.0f));  // Position of 14
    EXPECT_TRUE(approxEqual(gradInput.at({0, 0, 3, 3}), 1.0f));  // Position of 16
    
    // Other positions should be 0
    EXPECT_TRUE(approxEqual(gradInput.at({0, 0, 0, 0}), 0.0f));
}

TEST(MaxPooling2DLayerTest, InvalidInputRank) {
    MaxPooling2DLayer<float> poolLayer(2, 2);
    
    // 3D tensor instead of 4D
    Tensor<float> input({1, 4, 4});
    
    EXPECT_THROW(poolLayer.forward(input), std::runtime_error);
}

TEST(MaxPooling2DLayerTest, DifferentStride) {
    MaxPooling2DLayer<float> poolLayer(2, 1);  // 2x2 pool, stride 1
    
    // Input: 1 batch, 1 channel, 3x3 image
    Tensor<float> input({1, 1, 3, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });
    
    Tensor<float> output = poolLayer.forward(input);
    
    // With stride 1, output should be 2x2
    ASSERT_EQ(output.getShape(), Shape({1, 1, 2, 2}));
    
    EXPECT_TRUE(approxEqual(output.at({0, 0, 0, 0}), 5.0f));  // max(1,2,4,5)
    EXPECT_TRUE(approxEqual(output.at({0, 0, 0, 1}), 6.0f));  // max(2,3,5,6)
    EXPECT_TRUE(approxEqual(output.at({0, 0, 1, 0}), 8.0f));  // max(4,5,7,8)
    EXPECT_TRUE(approxEqual(output.at({0, 0, 1, 1}), 9.0f));  // max(5,6,8,9)
}

// ========== Flatten Layer Tests ==========

TEST(FlattenLayerTest, ForwardPass2D) {
    FlattenLayer<float> flattenLayer;
    
    // 2D input should remain unchanged
    Tensor<float> input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    Tensor<float> output = flattenLayer.forward(input);
    
    ASSERT_EQ(output.getShape(), input.getShape());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_TRUE(approxEqual(output.getData()[i], input.getData()[i]));
    }
}

TEST(FlattenLayerTest, ForwardPass3D) {
    FlattenLayer<float> flattenLayer;
    
    // 3D input: (batch=2, height=2, width=3)
    Tensor<float> input({2, 2, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    });
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Should flatten to (batch=2, features=6)
    ASSERT_EQ(output.getShape(), Shape({2, 6}));
    
    // Data should remain in same order
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_TRUE(approxEqual(output.getData()[i], input.getData()[i]));
    }
}

TEST(FlattenLayerTest, ForwardPass4D) {
    FlattenLayer<float> flattenLayer;
    
    // 4D input: (batch=1, channels=2, height=2, width=2)
    Tensor<float> input({1, 2, 2, 2}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    });
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Should flatten to (batch=1, features=8)
    ASSERT_EQ(output.getShape(), Shape({1, 8}));
    
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_TRUE(approxEqual(output.getData()[i], input.getData()[i]));
    }
}

TEST(FlattenLayerTest, BackwardPass) {
    FlattenLayer<float> flattenLayer;
    
    // Original 4D input
    Tensor<float> input({2, 2, 2, 2}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Gradient with flattened shape
    Tensor<float> gradOutput({2, 8}, {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f
    });
    
    Tensor<float> gradInput = flattenLayer.backward(gradOutput);
    
    // Should reshape back to original input shape
    ASSERT_EQ(gradInput.getShape(), input.getShape());
    
    // Data should be preserved
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_TRUE(approxEqual(gradInput.getData()[i], gradOutput.getData()[i]));
    }
}

TEST(FlattenLayerTest, InvalidInputRank) {
    FlattenLayer<float> flattenLayer;
    
    // 1D tensor is invalid
    Tensor<float> input({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    
    EXPECT_THROW(flattenLayer.forward(input), std::invalid_argument);
}

TEST(FlattenLayerTest, MultipleBatchSizes) {
    FlattenLayer<float> flattenLayer;
    
    // Test with different batch sizes
    Tensor<float> input1({3, 2, 2}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    });
    
    Tensor<float> output1 = flattenLayer.forward(input1);
    ASSERT_EQ(output1.getShape(), Shape({3, 4}));
    
    // Test with single batch
    Tensor<float> input2({1, 3, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });
    
    Tensor<float> output2 = flattenLayer.forward(input2);
    ASSERT_EQ(output2.getShape(), Shape({1, 9}));
}

} // namespace smart_dnn

#endif // TEST_POOLING_AND_FLATTEN_LAYERS_CPP

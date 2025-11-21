#ifndef TEST_FLATTEN_LAYER_CPP
#define TEST_FLATTEN_LAYER_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Layers/FlattenLayer.hpp"
#include "../utils/tensor_helpers.hpp"

namespace smart_dnn {

// ==================== FlattenLayer Forward Pass Tests ====================

TEST(FlattenLayerTest, Forward3DTo2D) {
    FlattenLayer<float> flattenLayer;
    
    // Create 3D input: (batch=2, height=3, width=4)
    Tensor<float> input({2, 3, 4}, {
        // Batch 1
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        // Batch 2
        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f
    });
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Output shape should be (2, 12)
    ASSERT_EQ(output.getShape(), Shape({2, 12}));
    ASSERT_EQ(output.getShape().size(), 24);
    
    // Verify data is preserved
    for (size_t i = 0; i < 24; ++i) {
        ASSERT_FLOAT_EQ(output.getData()[i], static_cast<float>(i + 1));
    }
}

TEST(FlattenLayerTest, Forward4DTo2D) {
    FlattenLayer<float> flattenLayer;
    
    // Create 4D input: (batch=2, channels=2, height=2, width=2)
    Tensor<float> input({2, 2, 2, 2});
    
    // Fill with sequential values
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        input.getData()[i] = static_cast<float>(i + 1);
    }
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Output shape should be (2, 8) - batch size 2, flattened size 8
    ASSERT_EQ(output.getShape(), Shape({2, 8}));
    ASSERT_EQ(output.getShape().size(), 16);
    
    // Verify data is preserved
    for (size_t i = 0; i < 16; ++i) {
        ASSERT_FLOAT_EQ(output.getData()[i], static_cast<float>(i + 1));
    }
}

TEST(FlattenLayerTest, Forward2DReturnsInput) {
    FlattenLayer<float> flattenLayer;
    
    // Create 2D input
    Tensor<float> input({3, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    });
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Output should be the same as input for 2D tensors
    ASSERT_EQ(output.getShape(), input.getShape());
    
    for (size_t i = 0; i < 12; ++i) {
        ASSERT_FLOAT_EQ(output.getData()[i], input.getData()[i]);
    }
}

TEST(FlattenLayerTest, ForwardSingleBatch) {
    FlattenLayer<float> flattenLayer;
    
    // Create input with batch size 1
    Tensor<float> input({1, 3, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    });
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Output shape should be (1, 12)
    ASSERT_EQ(output.getShape(), Shape({1, 12}));
    
    for (size_t i = 0; i < 12; ++i) {
        ASSERT_FLOAT_EQ(output.getData()[i], static_cast<float>(i + 1));
    }
}

TEST(FlattenLayerTest, ForwardLargeBatch) {
    FlattenLayer<float> flattenLayer;
    
    // Create input with larger batch size
    Tensor<float> input({10, 3, 4});
    
    // Fill with values
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        input.getData()[i] = static_cast<float>(i);
    }
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Output shape should be (10, 12)
    ASSERT_EQ(output.getShape(), Shape({10, 12}));
    ASSERT_EQ(output.getShape().size(), 120);
}

TEST(FlattenLayerTest, Forward1DInputThrows) {
    FlattenLayer<float> flattenLayer;
    
    // Create 1D input (invalid for flatten layer)
    Tensor<float> input({10}, 1.0f);
    
    // Should throw for 1D input
    EXPECT_THROW(flattenLayer.forward(input), std::invalid_argument);
}

// ==================== FlattenLayer Backward Pass Tests ====================

TEST(FlattenLayerTest, Backward3DShape) {
    FlattenLayer<float> flattenLayer;
    
    // Create 3D input and perform forward pass
    Tensor<float> input({2, 3, 4});
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        input.getData()[i] = static_cast<float>(i + 1);
    }
    
    Tensor<float> output = flattenLayer.forward(input);
    
    // Create gradient output (same shape as forward output)
    Tensor<float> gradOutput({2, 12});
    for (size_t i = 0; i < gradOutput.getShape().size(); ++i) {
        gradOutput.getData()[i] = static_cast<float>(i + 100);
    }
    
    // Perform backward pass
    Tensor<float> gradInput = flattenLayer.backward(gradOutput);
    
    // Gradient input should have the same shape as original input
    ASSERT_EQ(gradInput.getShape(), Shape({2, 3, 4}));
    
    // Verify data is preserved
    for (size_t i = 0; i < 24; ++i) {
        ASSERT_FLOAT_EQ(gradInput.getData()[i], static_cast<float>(i + 100));
    }
}

TEST(FlattenLayerTest, Backward4DShape) {
    FlattenLayer<float> flattenLayer;
    
    // Create 4D input and perform forward pass
    Tensor<float> input({2, 2, 2, 2});
    flattenLayer.forward(input);
    
    // Create gradient output
    Tensor<float> gradOutput({2, 8});
    for (size_t i = 0; i < gradOutput.getShape().size(); ++i) {
        gradOutput.getData()[i] = static_cast<float>(i * 2);
    }
    
    // Perform backward pass
    Tensor<float> gradInput = flattenLayer.backward(gradOutput);
    
    // Gradient input should have original 4D shape
    ASSERT_EQ(gradInput.getShape(), Shape({2, 2, 2, 2}));
    
    // Verify data is preserved
    for (size_t i = 0; i < 16; ++i) {
        ASSERT_FLOAT_EQ(gradInput.getData()[i], static_cast<float>(i * 2));
    }
}

TEST(FlattenLayerTest, BackwardPreservesGradients) {
    FlattenLayer<float> flattenLayer;
    
    // Forward pass
    Tensor<float> input({3, 4, 5});
    flattenLayer.forward(input);
    
    // Create specific gradient values
    Tensor<float> gradOutput({3, 20}, {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
        31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
        51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f
    });
    
    Tensor<float> gradInput = flattenLayer.backward(gradOutput);
    
    // Verify shape is restored
    ASSERT_EQ(gradInput.getShape(), Shape({3, 4, 5}));
    
    // Verify all gradient values are preserved
    for (size_t i = 0; i < 60; ++i) {
        ASSERT_FLOAT_EQ(gradInput.getData()[i], static_cast<float>(i + 1));
    }
}

TEST(FlattenLayerTest, ForwardBackwardRoundTrip) {
    FlattenLayer<float> flattenLayer;
    
    // Create input
    Tensor<float> input({2, 3, 4});
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        input.getData()[i] = static_cast<float>(i);
    }
    
    // Forward pass
    Tensor<float> output = flattenLayer.forward(input);
    ASSERT_EQ(output.getShape(), Shape({2, 12}));
    
    // Create gradient with same values as output
    Tensor<float> gradOutput({2, 12});
    for (size_t i = 0; i < gradOutput.getShape().size(); ++i) {
        gradOutput.getData()[i] = output.getData()[i];
    }
    
    // Backward pass
    Tensor<float> gradInput = flattenLayer.backward(gradOutput);
    
    // Gradient should have same shape as input
    ASSERT_EQ(gradInput.getShape(), input.getShape());
    
    // Gradient values should match input values (since we copied them)
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        ASSERT_FLOAT_EQ(gradInput.getData()[i], input.getData()[i]);
    }
}

} // namespace smart_dnn

#endif // TEST_FLATTEN_LAYER_CPP

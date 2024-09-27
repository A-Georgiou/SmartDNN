#ifndef TEST_MAX_POOLING_CPP
#define TEST_MAX_POOLING_CPP

#include <gtest/gtest.h>
#include "tests/utils/tensor_helpers.hpp"
#include "smart_dnn/optimizers/AdamOptimizer.hpp"
#include "smart_dnn/regularisation/MaxPooling2DLayer.hpp"  

namespace sdnn {


TEST(MaxPooling2DLayerTest, ForwardPass2x2) {
    MaxPooling2DLayer maxPoolLayer(2, 2);  // 2x2 pool size, stride 2

    // Create a (1, 1, 4, 4) input tensor (batch_size = 1, channels = 1, height = 4, width = 4)
    Tensor input({1, 1, 4, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });

    Tensor output = maxPoolLayer.forward(input);

    ASSERT_EQ(output.shape(), Shape({1, 1, 2, 2}));

    std::vector<float> expected_values = {6.0f, 8.0f, 14.0f, 16.0f};
    for (size_t i = 0; i < output.shape().size(); ++i) {
        ASSERT_EQ(output.at<float>({0, 0, i / 2, i % 2}), expected_values[i]) 
            << "Output mismatch at index " << i;
    }
}

TEST(MaxPooling2DLayerTest, ForwardPassWithChannels) {
    MaxPooling2DLayer maxPoolLayer(2, 2);  // 2x2 pool size, stride 2

    // Create a (1, 2, 4, 4) input tensor (batch_size = 1, channels = 2, height = 4, width = 4)
    Tensor input({1, 2, 4, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
        
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f,
        29.0f, 30.0f, 31.0f, 32.0f
    });

    Tensor output = maxPoolLayer.forward(input);

    ASSERT_EQ(output.shape(), Shape({1, 2, 2, 2}));

    std::vector<float> expected_values = {6.0f, 8.0f, 14.0f, 16.0f, 22.0f, 24.0f, 30.0f, 32.0f};
    for (size_t c = 0; c < 2; ++c) {
        for (size_t i = 0; i < 4; ++i) {
            ASSERT_EQ(output.at<float>({0, c, i / 2, i % 2}), expected_values[c * 4 + i]) 
                << "Output mismatch at channel " << c << ", index " << i;
        }
    }
}

TEST(MaxPooling2DLayerTest, BackwardPass) {
    MaxPooling2DLayer maxPoolLayer(2, 2);  // 2x2 pool size, stride 2

    // Input tensor (1, 1, 4, 4)
    Tensor input({1, 1, 4, 4}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });

    maxPoolLayer.forward(input);

    // Gradient of the output (1, 1, 2, 2)
    Tensor gradOutput({1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});

    Tensor gradInput = maxPoolLayer.backward(gradOutput);

    ASSERT_EQ(gradInput.shape(), Shape({1, 1, 4, 4}));

    std::vector<float> expected_grad = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f
    };

    for (size_t i = 0; i < gradInput.shape().size(); ++i) {
        ASSERT_EQ(gradInput.at<float>({0, 0, i / 4, i % 4}), expected_grad[i]) 
            << "Gradient mismatch at index " << i;
    }
}

TEST(MaxPooling2DLayerTest, ForwardPassWithStride) {
    MaxPooling2DLayer maxPoolLayer(2, 1);  // 2x2 pool size, stride 1

    // Create a (1, 1, 3, 3) input tensor
    Tensor input({1, 1, 3, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });

    Tensor output = maxPoolLayer.forward(input);

    ASSERT_EQ(output.shape(), Shape({1, 1, 2, 2}));

    std::vector<float> expected_values = {5.0f, 6.0f, 8.0f, 9.0f};
    for (size_t i = 0; i < output.shape().size(); ++i) {
        ASSERT_EQ(output.at<float>({0, 0, i / 2, i % 2}), expected_values[i]) 
            << "Output mismatch at index " << i;
    }
}

TEST(MaxPooling2DLayerTest, ForwardPassWithPadding) {
    MaxPooling2DLayer maxPoolLayer(2, 2, 1);  // 2x2 pool size, stride 2, padding 1

    // Create a (1, 1, 3, 3) input tensor
    Tensor input({1, 1, 3, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    });

    Tensor output = maxPoolLayer.forward(input);

    std::cout << "output: " << output.toString() << std::endl;
    
    std::vector<float> expected_values = {1.0f, 3.0f, 7.0f, 9.0f};
    for (size_t i = 0; i < output.shape().size(); ++i) {
        ASSERT_EQ(output.at<float>({0, 0, i / 2, i % 2}), expected_values[i]) 
            << "Output mismatch at index " << i;
    }
}




}

#endif  // TEST_MAX_POOLING_CPP
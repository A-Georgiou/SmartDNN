#ifndef TEST_BATCH_NORM_CPP
#define TEST_BATCH_NORM_CPP

#include <gtest/gtest.h>
#include "tests/utils/tensor_helpers.hpp"
#include "smart_dnn/optimizers/AdamOptimizer.hpp"
#include "smart_dnn/regularisation/BatchNormalizationLayer.hpp"

namespace sdnn {

TEST(BatchNormalizationLayerTest, ForwardPassTrainingMode2D) {
    BatchNormalizationLayer bnLayer(3);  // 3 features (channels)

    // Create a (2, 3) input tensor (batch_size = 2, features = 3)
    Tensor input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});  // [[1, 2, 3], [4, 5, 6]]

    Tensor output = bnLayer.forward(input);

    ASSERT_EQ(output.shape(), Shape({2, 3}));

    Tensor out_mean = mean(output, {0});
    Tensor out_variance = variance(output, out_mean, {0});
    for (size_t i = 0; i < out_mean.shape().size(); ++i) {
        ASSERT_NEAR(out_mean.at<float>(i), 0.0f, 1e-5) << "Mean mismatch at index " << i;
        ASSERT_NEAR(out_variance.at<float>(i), 1.0f, 1e-5) << "Variance mismatch at index " << i;
    }
}

TEST(BatchNormalizationLayerTest, ForwardPassTrainingMode4D) {
    BatchNormalizationLayer bnLayer(3);  // 3 features (channels)

    // Create a (2, 3, 2, 2) input tensor (batch_size = 2, channels = 3, height = 2, width = 2)
    Tensor input({2, 3, 2, 2}, {
        1.0f, 2.0f, 3.0f, 4.0f,  // First batch, channel 1
        5.0f, 6.0f, 7.0f, 8.0f,  // First batch, channel 2
        9.0f, 10.0f, 11.0f, 12.0f, // First batch, channel 3
        13.0f, 14.0f, 15.0f, 16.0f,  // Second batch, channel 1
        17.0f, 18.0f, 19.0f, 20.0f,  // Second batch, channel 2
        21.0f, 22.0f, 23.0f, 24.0f   // Second batch, channel 3
    });

    // Perform forward pass
    Tensor output = bnLayer.forward(input);
    
    std::vector<float> expected_values = {-1.5741f, -1.3634f, -1.1528f, -0.9422f, 0.9422f, 1.1528f, 1.3634f, 1.5741f};

    // Check output shape
    ASSERT_EQ(output.shape(), Shape({2, 3, 2, 2}));
    Tensor out_mean = mean(output, {0, 2, 3});

    std::vector<float> expected_mean = {0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < out_mean.shape().size(); ++i) {
        ASSERT_NEAR(out_mean.at<float>(i), expected_mean[i], 1e-5) << "Mean mismatch at index " << i;
    }
    
    Tensor reshaped_mean = reshape(out_mean, Shape({1, 3, 1, 1}));
    Tensor out_variance = variance(output, reshaped_mean, {0, 2, 3});
    std::vector<float> expected_variance = {1.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < out_variance.shape().size(); ++i) {
        ASSERT_NEAR(out_variance.at<float>(i), expected_variance[i], 1e-5) << "Variance mismatch at index " << i;
    }
}

TEST(BatchNormalizationLayerTest, BackwardPass2D) {
    BatchNormalizationLayer bnLayer(3);  // 3 features (channels)

    Tensor input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});  // [[1, 2, 3], [4, 5, 6]]

    bnLayer.forward(input);

    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});  // All ones

    Tensor gradInput = bnLayer.backward(gradOutput);

    ASSERT_EQ(gradInput.shape(), Shape({2, 3}));

    for (size_t i = 0; i < gradInput.shape().size(); ++i) {
        ASSERT_TRUE(std::isfinite(gradInput.at<float>(i))) << "Backward pass produced non-finite value at index " << i;
    }
}

TEST(BatchNormalizationLayerTest, UpdateWeights) {
    BatchNormalizationLayer bnLayer(3);  // 3 features (channels)

    Tensor input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    bnLayer.forward(input);
    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    bnLayer.backward(gradOutput);

    AdamOptimizer optimizer;

    ASSERT_NO_THROW(bnLayer.updateWeights(optimizer));
}


}

#endif  // TEST_BATCH_NORM_CPP
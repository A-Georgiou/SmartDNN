#ifndef TEST_DROPOUT_CPP
#define TEST_DROPOUT_CPP

#include <gtest/gtest.h>
#include "smart_dnn/tensor/TensorBase.hpp"
#include "tests/utils/tensor_helpers.hpp"
#include "smart_dnn/regularisation/DropoutLayer.hpp"

namespace sdnn {


TEST(DropoutLayerTest, ForwardPassTrainingMode) {
    DropoutLayer dropoutLayer(0.5);
    dropoutLayer.setTrainingMode(true);

    Tensor input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor output = dropoutLayer.forward(input);

    ASSERT_EQ(output.shape(), Shape({2, 3}));
    
    // Check that some elements are zeroed out and others are scaled
    bool hasZero = false;
    bool hasScaled = false;
    for (size_t i = 0; i < output.shape().size(); ++i) {
        float val = output.at<float>(i);
        if (val == 0) hasZero = true;
        if (std::abs(val - input.at<float>(i) * 2) < 1e-5) hasScaled = true;
    }

    EXPECT_TRUE(hasZero) << "No elements were dropped out";
    EXPECT_TRUE(hasScaled) << "No elements were correctly scaled";
}

TEST(DropoutLayerTest, ForwardPassInferenceMode) {
    DropoutLayer dropoutLayer(0.5);
    dropoutLayer.setTrainingMode(false);

    Tensor input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor output = dropoutLayer.forward(input);

    ASSERT_EQ(output.shape(), Shape({2, 3}));
    
    // In inference mode, output should be identical to input
    for (size_t i = 0; i < output.shape().size(); ++i) {
        EXPECT_NEAR(output.at<float>(i), input.at<float>(i), 1e-5);
    }
}

TEST(DropoutLayerTest, BackwardPassTrainingMode) {
    DropoutLayer dropoutLayer(0.5);
    dropoutLayer.setTrainingMode(true);

    Tensor input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor forwardOutput = dropoutLayer.forward(input);

    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gradInput = dropoutLayer.backward(gradOutput);

    ASSERT_EQ(gradInput.shape(), Shape({2, 3}));

    // Check that gradient is zero where forward output was zero, and scaled where it wasn't
    for (size_t i = 0; i < gradInput.shape().size(); ++i) {
        if (forwardOutput.at<float>(i) == 0) {
            EXPECT_NEAR(gradInput.at<float>(i), 0, 1e-5);
        } else {
            EXPECT_NEAR(gradInput.at<float>(i), 2, 1e-5);
        }
    }
}

TEST(DropoutLayerTest, BackwardPassInferenceMode) {
    DropoutLayer dropoutLayer(0.5);
    dropoutLayer.setTrainingMode(false);

    Tensor input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    dropoutLayer.forward(input);

    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gradInput = dropoutLayer.backward(gradOutput);

    ASSERT_EQ(gradInput.shape(), Shape({2, 3}));

    // In inference mode, gradient should be identical to gradOutput
    for (size_t i = 0; i < gradInput.shape().size(); ++i) {
        EXPECT_NEAR(gradInput.at<float>(i), gradOutput.at<float>(i), 1e-5);
    }
}

}   // namespace sdnn

#endif

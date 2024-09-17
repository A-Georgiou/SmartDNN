#ifndef TEST_RELU_CPP
#define TEST_RELU_CPP

#include <gtest/gtest.h>
#include "smart_dnn/activations/ReLU.hpp"
#include <cmath>

namespace sdnn {
// Helper function to check if two floats are approximately equal
bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}


TEST(ReLUTest, ForwardPass) {
    ReLU relu;
    Tensor input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor output = relu.forward(input);

    ASSERT_EQ(output.shape(), input.shape());
    EXPECT_TRUE(approxEqual(output.at<float>(0), 0.0f));
    EXPECT_TRUE(approxEqual(output.at<float>(1), 0.0f));
    EXPECT_TRUE(approxEqual(output.at<float>(2), 0.0f));
    EXPECT_TRUE(approxEqual(output.at<float>(3), 1.0f));
    EXPECT_TRUE(approxEqual(output.at<float>(4), 2.0f));
    EXPECT_TRUE(approxEqual(output.at<float>(5), 3.0f));
}

TEST(ReLUTest, BackwardPass) {
    ReLU relu;
    Tensor input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gradInput = relu.backward(input, gradOutput);

    ASSERT_EQ(gradInput.shape(), input.shape());
    EXPECT_TRUE(approxEqual(gradInput.at<float>(0), 0.0f));
    EXPECT_TRUE(approxEqual(gradInput.at<float>(1), 0.0f));
    EXPECT_TRUE(approxEqual(gradInput.at<float>(2), 0.0f));
    EXPECT_TRUE(approxEqual(gradInput.at<float>(3), 1.0f));
    EXPECT_TRUE(approxEqual(gradInput.at<float>(4), 1.0f));
    EXPECT_TRUE(approxEqual(gradInput.at<float>(5), 1.0f));
}


} // namespace sdnn

#endif // TEST_RELU_CPP
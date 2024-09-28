#ifndef TEST_ACTIVATIONS_CPP
#define TEST_ACTIVATIONS_CPP

#include <gtest/gtest.h>
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/activations/ReLU.hpp"
#include "smart_dnn/activations/LeakyReLU.hpp"
#include "smart_dnn/activations/Sigmoid.hpp"
#include "smart_dnn/activations/Tanh.hpp"
#include "smart_dnn/activations/Softmax.hpp"
#include <cmath>

namespace sdnn {
// Helper function to check if two floats are approximately equal
bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

TEST(LeakyReLUTest, ForwardPass) {
    LeakyReLU leakyRelu(0.1f);
    Tensor input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor output = leakyRelu.forward(input);

    ASSERT_EQ(output.shape(), input.shape());
    EXPECT_TRUE(approxEqual(output.at<float>(0), -0.2f));
    EXPECT_TRUE(approxEqual(output.at<float>(1), -0.1f));
    EXPECT_TRUE(approxEqual(output.at<float>(2), 0.0f));
    EXPECT_TRUE(approxEqual(output.at<float>(3), 1.0f));
    EXPECT_TRUE(approxEqual(output.at<float>(4), 2.0f));
    EXPECT_TRUE(approxEqual(output.at<float>(5), 3.0f));
}

TEST(LeakyReLUTest, BackwardPass) {
    LeakyReLU leakyRelu(0.1f);
    Tensor input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gradInput = leakyRelu.backward(input, gradOutput);

    ASSERT_EQ(gradInput.shape(), input.shape());
    EXPECT_TRUE(approxEqual(gradInput.at<float>(0), 0.1f));
    EXPECT_TRUE(approxEqual(gradInput.at<float>(1), 0.1f));
    EXPECT_TRUE(approxEqual(gradInput.at<float>(2), 0.1f));
    EXPECT_TRUE(approxEqual(gradInput.at<float>(3), 1.0f));
    EXPECT_TRUE(approxEqual(gradInput.at<float>(4), 1.0f));
    EXPECT_TRUE(approxEqual(gradInput.at<float>(5), 1.0f));
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

TEST(SigmoidTest, ForwardPass) {
    Sigmoid sigmoid;
    Tensor input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor output = sigmoid.forward(input);

    ASSERT_EQ(output.shape(), input.shape());
    EXPECT_TRUE(approxEqual(output.at<float>(0), 1.0f / (1.0f + std::exp(2.0f))));
    EXPECT_TRUE(approxEqual(output.at<float>(1), 1.0f / (1.0f + std::exp(1.0f))));
    EXPECT_TRUE(approxEqual(output.at<float>(2), 0.5f));
    EXPECT_TRUE(approxEqual(output.at<float>(3), 1.0f / (1.0f + std::exp(-1.0f))));
    EXPECT_TRUE(approxEqual(output.at<float>(4), 1.0f / (1.0f + std::exp(-2.0f))));
    EXPECT_TRUE(approxEqual(output.at<float>(5), 1.0f / (1.0f + std::exp(-3.0f))));
}

TEST(SigmoidTest, BackwardPass) {
    Sigmoid sigmoid;
    Tensor input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gradInput = sigmoid.backward(input, gradOutput);

    ASSERT_EQ(gradInput.shape(), input.shape());
    for (size_t i = 0; i < input.shape().size(); ++i) {
        float s = 1.0f / (1.0f + std::exp(-input.at<float>(i)));
        float out = s * (1 - s);
        EXPECT_TRUE(approxEqual(gradInput.at<float>(i), out));
    }
}

TEST(TanhTest, ForwardPass) {
    Tanh tanh;
    Tensor input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor output = tanh.forward(input);

    ASSERT_EQ(output.shape(), input.shape());
    for (size_t i = 0; i < input.shape().size(); ++i) {
        EXPECT_TRUE(approxEqual(output.at<float>(i), std::tanh(input.at<float>(i))));
    }
}

TEST(TanhTest, BackwardPass) {
    Tanh tanh;
    Tensor input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gradInput = tanh.backward(input, gradOutput);

    ASSERT_EQ(gradInput.shape(), input.shape());
    for (size_t i = 0; i < input.shape().size(); ++i) {
        float t = std::tanh(input.at<float>(i));
        EXPECT_TRUE(approxEqual(gradInput.at<float>(i), 1.0f - t * t));
    }
}

TEST(SoftmaxTest, ForwardPass) {
    Softmax softmax;
    Tensor input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor output = softmax.forward(input);

    ASSERT_EQ(output.shape(), input.shape());
    float sum1 = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
    float sum2 = std::exp(4.0f) + std::exp(5.0f) + std::exp(6.0f);
    EXPECT_TRUE(approxEqual(output.at<float>(0), std::exp(1.0f) / sum1));
    EXPECT_TRUE(approxEqual(output.at<float>(1), std::exp(2.0f) / sum1));
    EXPECT_TRUE(approxEqual(output.at<float>(2), std::exp(3.0f) / sum1));
    EXPECT_TRUE(approxEqual(output.at<float>(3), std::exp(4.0f) / sum2));
    EXPECT_TRUE(approxEqual(output.at<float>(4), std::exp(5.0f) / sum2));
    EXPECT_TRUE(approxEqual(output.at<float>(5), std::exp(6.0f) / sum2));
}

TEST(SoftmaxTest, BackwardPass) {
    Softmax  softmax;
    Tensor input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor gradOutput({2, 3}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    Tensor gradInput = softmax.backward(input, gradOutput);

    ASSERT_EQ(gradInput.shape(), input.shape());
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < 3; ++i) {
        EXPECT_NE(gradInput.at<float>(i), 0.0f);
        sum1 += gradInput.at<float>(i);
        EXPECT_NE(gradInput.at<float>(i+3), 0.0f);
        sum2 += gradInput.at<float>(i+3);
    }
    EXPECT_TRUE(approxEqual(sum1, 0.0f));
    EXPECT_TRUE(approxEqual(sum2, 0.0f));
}

/*

TEST SOFTMAX CAN HANDLE BATCHES

*/
TEST(SoftmaxTest, ForwardPassBatch) {
    Softmax softmax;
    // 3x2x2 tensor (3 batches, 2x2 for each batch)
    Tensor input({3, 1, 2, 2}, {
        1.0f, 2.0f, 3.0f, 4.0f,     // Batch 1
        0.5f, 1.5f, 2.5f, 3.5f,     // Batch 2
        -1.0f, 0.0f, 1.0f, 2.0f     // Batch 3
    });
    Tensor output = softmax.forward(input);

    ASSERT_EQ(output.shape(), input.shape());

    auto computeSoftmax = [](const std::vector<float>& batch) {
        float maxVal = *std::max_element(batch.begin(), batch.end());
        std::vector<float> expVals(batch.size());
        float sumExp = 0.0f;
        for (size_t i = 0; i < batch.size(); ++i) {
            expVals[i] = std::exp(batch[i] - maxVal);
            sumExp += expVals[i];
        }
        for (size_t i = 0; i < batch.size(); ++i) {
            expVals[i] /= sumExp;
        }
        return expVals;
    };

    // Check each batch
    for (int b = 0; b < 3; ++b) {
        std::vector<float> batchInput = {
            input.at<float>(b * 4), input.at<float>(b * 4 + 1),
            input.at<float>(b * 4 + 2), input.at<float>(b * 4 + 3)
        };
        std::vector<float> expectedOutput = computeSoftmax(batchInput);

        for (int i = 0; i < 4; ++i) {
            EXPECT_TRUE(approxEqual(output.at<float>(b * 4 + i), expectedOutput[i]))
                << "Mismatch at batch " << b << ", index " << i;
        }
    }
}

TEST(SoftmaxTest, BackwardPassBatch) {
    Softmax softmax;
    // 3x2x2 tensor (3 batches, 2x2 for each batch)
    Tensor input({3, 1, 2, 2}, {
        1.0f, 2.0f, 3.0f, 4.0f,     // Batch 1
        0.5f, 1.5f, 2.5f, 3.5f,     // Batch 2
        -1.0f, 0.0f, 1.0f, 2.0f     // Batch 3
    });
    Tensor gradOutput({3, 1, 2, 2}, {
        0.1f, 0.2f, 0.3f, 0.4f,     // Batch 1
        0.5f, 0.6f, 0.7f, 0.8f,     // Batch 2
        0.9f, 1.0f, 1.1f, 1.2f      // Batch 3
    });
    Tensor gradInput = softmax.backward(input, gradOutput);

    ASSERT_EQ(gradInput.shape(), input.shape());

    // Check each batch
    for (int b = 0; b < 3; ++b) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            EXPECT_NE(gradInput.at<float>(b * 4 + i), 0.0f)
                << "Gradient should not be zero at batch " << b << ", index " << i;
            sum += gradInput.at<float>(b * 4 + i);
        }
        EXPECT_TRUE(approxEqual(sum, 0.0f))
            << "Sum of gradients for batch " << b << " should be close to zero";
    }
}

} // namespace sdnn

#endif // TEST_ACTIVATIONS_CPP
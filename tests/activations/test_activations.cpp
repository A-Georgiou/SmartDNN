#ifndef TEST_ACTIVATIONS_CPP
#define TEST_ACTIVATIONS_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Activations/ReLU.hpp"
#include "../../smart_dnn/Activations/LeakyReLU.hpp"
#include "../../smart_dnn/Activations/Sigmoid.hpp"
#include "../../smart_dnn/Activations/Softmax.hpp"
#include "../../smart_dnn/Activations/Tanh.hpp"
#include "../../smart_dnn/Activations/Swish.hpp"
#include "../../smart_dnn/Activations/Mish.hpp"
#include <cmath>

namespace smart_dnn {

// Helper function to check if two floats are approximately equal
bool approxEqual(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

TEST(LeakyReLUTest, ForwardPass) {
    LeakyReLU<float> leakyRelu(0.1f);
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> output = leakyRelu.forward(input);

    ASSERT_EQ(output.getShape(), input.getShape());
    EXPECT_TRUE(approxEqual(output.getData()[0], -0.2f));
    EXPECT_TRUE(approxEqual(output.getData()[1], -0.1f));
    EXPECT_TRUE(approxEqual(output.getData()[2], 0.0f));
    EXPECT_TRUE(approxEqual(output.getData()[3], 1.0f));
    EXPECT_TRUE(approxEqual(output.getData()[4], 2.0f));
    EXPECT_TRUE(approxEqual(output.getData()[5], 3.0f));
}

TEST(LeakyReLUTest, BackwardPass) {
    LeakyReLU<float> leakyRelu(0.1f);
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor<float> gradInput = leakyRelu.backward(input, gradOutput);

    std::cout << gradInput.toDataString() << std::endl;

    ASSERT_EQ(gradInput.getShape(), input.getShape());
    EXPECT_TRUE(approxEqual(gradInput.getData()[0], 0.1f));
    EXPECT_TRUE(approxEqual(gradInput.getData()[1], 0.1f));
    EXPECT_TRUE(approxEqual(gradInput.getData()[2], 1.0f));
    EXPECT_TRUE(approxEqual(gradInput.getData()[3], 1.0f));
    EXPECT_TRUE(approxEqual(gradInput.getData()[4], 1.0f));
    EXPECT_TRUE(approxEqual(gradInput.getData()[5], 1.0f));
}

TEST(ReLUTest, ForwardPass) {
    ReLU<float> relu;
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> output = relu.forward(input);

    ASSERT_EQ(output.getShape(), input.getShape());
    EXPECT_TRUE(approxEqual(output.getData()[0], 0.0f));
    EXPECT_TRUE(approxEqual(output.getData()[1], 0.0f));
    EXPECT_TRUE(approxEqual(output.getData()[2], 0.0f));
    EXPECT_TRUE(approxEqual(output.getData()[3], 1.0f));
    EXPECT_TRUE(approxEqual(output.getData()[4], 2.0f));
    EXPECT_TRUE(approxEqual(output.getData()[5], 3.0f));
}

TEST(ReLUTest, BackwardPass) {
    ReLU<float> relu;
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor<float> gradInput = relu.backward(input, gradOutput);

    ASSERT_EQ(gradInput.getShape(), input.getShape());
    EXPECT_TRUE(approxEqual(gradInput.getData()[0], 0.0f));
    EXPECT_TRUE(approxEqual(gradInput.getData()[1], 0.0f));
    EXPECT_TRUE(approxEqual(gradInput.getData()[2], 0.0f));
    EXPECT_TRUE(approxEqual(gradInput.getData()[3], 1.0f));
    EXPECT_TRUE(approxEqual(gradInput.getData()[4], 1.0f));
    EXPECT_TRUE(approxEqual(gradInput.getData()[5], 1.0f));
}

TEST(SigmoidTest, ForwardPass) {
    Sigmoid<float> sigmoid;
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> output = sigmoid.forward(input);

    ASSERT_EQ(output.getShape(), input.getShape());
    EXPECT_TRUE(approxEqual(output.getData()[0], 1.0f / (1.0f + std::exp(2.0f))));
    EXPECT_TRUE(approxEqual(output.getData()[1], 1.0f / (1.0f + std::exp(1.0f))));
    EXPECT_TRUE(approxEqual(output.getData()[2], 0.5f));
    EXPECT_TRUE(approxEqual(output.getData()[3], 1.0f / (1.0f + std::exp(-1.0f))));
    EXPECT_TRUE(approxEqual(output.getData()[4], 1.0f / (1.0f + std::exp(-2.0f))));
    EXPECT_TRUE(approxEqual(output.getData()[5], 1.0f / (1.0f + std::exp(-3.0f))));
}

TEST(SigmoidTest, BackwardPass) {
    Sigmoid<float> sigmoid;
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor<float> gradInput = sigmoid.backward(input, gradOutput);

    ASSERT_EQ(gradInput.getShape(), input.getShape());
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        float s = 1.0f / (1.0f + std::exp(-input.getData()[i]));
        EXPECT_TRUE(approxEqual(gradInput.getData()[i], s * (1 - s)));
    }
}

TEST(SigmoidTest, DoublePrecisionReferencesAndGradientScaling) {
    Sigmoid<double> sigmoid;
    Tensor<double> input({1, 3}, {1.0, -1.0, 1.00000001});
    Tensor<double> gradOutput({1, 3}, {2.5, 2.5, 2.5});
    Tensor<double> output = sigmoid.forward(input);
    Tensor<double> gradInput = sigmoid.backward(input, gradOutput);
    const long double values[] = {1.0L, -1.0L, 1.00000001L};

    ASSERT_EQ(output.getShape(), input.getShape());
    ASSERT_EQ(gradInput.getShape(), input.getShape());
    for (size_t i = 0; i < 3; ++i) {
        const long double exponential = std::exp(-values[i]);
        const long double denominator = 1.0L + exponential;
        const long double expectedForward = 1.0L / denominator;
        const long double expectedDerivative = exponential / (denominator * denominator);
        const long double actualForward = static_cast<long double>(output.getData()[i]);
        const long double actualGradient = static_cast<long double>(gradInput.getData()[i]);

        EXPECT_LT(std::abs(actualForward - expectedForward) / std::abs(expectedForward), 1e-12L);
        EXPECT_LT(std::abs(actualGradient - expectedDerivative * 2.5L) /
                      std::abs(expectedDerivative * 2.5L),
                  1e-12L);
    }

    EXPECT_NE(output.getData()[0], output.getData()[2]);
    EXPECT_NE(gradInput.getData()[0], gradInput.getData()[2]);
}

TEST(SigmoidTest, DoubleCentralDifferenceMatchesBackward) {
    Sigmoid<double> sigmoid;
    const double values[] = {1.0, -1.0, 1.00000001};
    const double upstream = 2.5;
    const double h = 1e-5;
    Tensor<double> input({1, 3}, {values[0], values[1], values[2]});
    Tensor<double> gradOutput({1, 3}, {upstream, upstream, upstream});
    Tensor<double> gradInput = sigmoid.backward(input, gradOutput);

    ASSERT_EQ(gradInput.getShape(), input.getShape());
    for (size_t i = 0; i < 3; ++i) {
        Tensor<double> plusInput(Shape{1}, {values[i] + h});
        Tensor<double> minusInput(Shape{1}, {values[i] - h});
        const double numerical = (sigmoid.forward(plusInput).getData()[0] -
                                  sigmoid.forward(minusInput).getData()[0]) /
                                 (2.0 * h) * upstream;
        const double analytical = gradInput.getData()[i];
        EXPECT_LT(std::abs(numerical - analytical) /
                      (std::abs(numerical) + std::abs(analytical)),
                  1e-4);
    }
}

TEST(SoftmaxTest, ForwardPass) {
    Softmax<float> softmax;
    Tensor<float> input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> output = softmax.forward(input);

    ASSERT_EQ(output.getShape(), input.getShape());
    float sum1 = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
    float sum2 = std::exp(4.0f) + std::exp(5.0f) + std::exp(6.0f);
    EXPECT_TRUE(approxEqual(output.getData()[0], std::exp(1.0f) / sum1));
    EXPECT_TRUE(approxEqual(output.getData()[1], std::exp(2.0f) / sum1));
    EXPECT_TRUE(approxEqual(output.getData()[2], std::exp(3.0f) / sum1));
    EXPECT_TRUE(approxEqual(output.getData()[3], std::exp(4.0f) / sum2));
    EXPECT_TRUE(approxEqual(output.getData()[4], std::exp(5.0f) / sum2));
    EXPECT_TRUE(approxEqual(output.getData()[5], std::exp(6.0f) / sum2));
}

TEST(SoftmaxTest, BackwardPass) {
    Softmax<float> softmax;
    Tensor<float> input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor<float> gradOutput({2, 3}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    Tensor<float> gradInput = softmax.backward(input, gradOutput);

    ASSERT_EQ(gradInput.getShape(), input.getShape());
    // The exact values for backward pass are complex to calculate by hand,
    // so we'll just check that the gradients are non-zero and sum to zero for each sample
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < 3; ++i) {
        EXPECT_NE(gradInput.getData()[i], 0.0f);
        sum1 += gradInput.getData()[i];
        EXPECT_NE(gradInput.getData()[i+3], 0.0f);
        sum2 += gradInput.getData()[i+3];
    }
    EXPECT_TRUE(approxEqual(sum1, 0.0f));
    EXPECT_TRUE(approxEqual(sum2, 0.0f));
}

TEST(TanhTest, ForwardPass) {
    Tanh<float> tanh;
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> output = tanh.forward(input);

    ASSERT_EQ(output.getShape(), input.getShape());
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        EXPECT_TRUE(approxEqual(output.getData()[i], std::tanh(input.getData()[i])));
    }
}

TEST(TanhTest, BackwardPass) {
    Tanh<float> tanh;
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor<float> gradInput = tanh.backward(input, gradOutput);

    ASSERT_EQ(gradInput.getShape(), input.getShape());
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        float t = std::tanh(input.getData()[i]);
        EXPECT_TRUE(approxEqual(gradInput.getData()[i], 1.0f - t * t));
    }
}

TEST(SwishTest, ForwardPass) {
    Swish<float> swish;
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> output = swish.forward(input);

    ASSERT_EQ(output.getShape(), input.getShape());
    
    // Test expected values: f(x) = x * sigmoid(x)
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        float x = input.getData()[i];
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
        float expected = x * sigmoid_x;
        EXPECT_TRUE(approxEqual(output.getData()[i], expected));
    }
}

TEST(SwishTest, BackwardPass) {
    Swish<float> swish;
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor<float> gradInput = swish.backward(input, gradOutput);

    ASSERT_EQ(gradInput.getShape(), input.getShape());
    
    // Test expected derivative: f'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        float x = input.getData()[i];
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
        float expected = sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));
        EXPECT_TRUE(approxEqual(gradInput.getData()[i], expected));
    }
}

TEST(MishTest, ForwardPass) {
    Mish<float> mish;
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> output = mish.forward(input);

    ASSERT_EQ(output.getShape(), input.getShape());
    
    // Test expected values: f(x) = x * tanh(softplus(x))
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        float x = input.getData()[i];
        float softplus_x = std::log(1.0f + std::exp(x));
        float expected = x * std::tanh(softplus_x);
        EXPECT_TRUE(approxEqual(output.getData()[i], expected));
    }
}

TEST(MishTest, BackwardPass) {
    Mish<float> mish;
    Tensor<float> input({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor<float> gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor<float> gradInput = mish.backward(input, gradOutput);

    ASSERT_EQ(gradInput.getShape(), input.getShape());
    
    // Test expected derivative
    for (size_t i = 0; i < input.getShape().size(); ++i) {
        float x = input.getData()[i];
        float softplus_x = std::log(1.0f + std::exp(x));
        float tanh_softplus = std::tanh(softplus_x);
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
        float sech_squared = 1.0f - tanh_softplus * tanh_softplus;
        float expected = tanh_softplus + x * sech_squared * sigmoid_x;
        EXPECT_TRUE(approxEqual(gradInput.getData()[i], expected));
    }
}

} // namespace smart_dnn

#endif // TEST_ACTIVATIONS_CPP
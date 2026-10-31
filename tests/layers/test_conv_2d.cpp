
#ifndef TEST_CONV_2D_CPP
#define TEST_CONV_2D_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Layers/Conv2DLayer.hpp"
#include "../utils/tensor_helpers.hpp"
#include "../../smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "../../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../../smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "../../smart_dnn/Regularisation/DropoutLayer.hpp"

namespace smart_dnn {

TEST(Conv2DLayerTest, ForwardPassHardcodedConv2D) {
    Conv2DLayer<float> convLayer(1, 1, 3); // 1 input channel, 1 output channel, 3x3 kernel

    // Set weights and biases
    Tensor<float> weightValues({1, 1, 3, 3}, {1, 0, -1,
                                              0, 1, 0,
                                              -1, 0, 1});
    Tensor<float> biasValues({1, 1}, {0.1f});

    convLayer.setWeights(weightValues);
    convLayer.setBiases(biasValues);

    // Create an input tensor (1 batch, 1 channel, 5x5 image)
    Tensor<float> input({1, 1, 5, 5}, {1, 2, 3, 4, 5,
                                       2, 3, 4, 5, 6,
                                       3, 4, 5, 6, 7,
                                       4, 5, 6, 7, 8,
                                       5, 6, 7, 8, 9});

    // Perform forward pass
    Tensor<float> output = convLayer.forward(input);

    // Expected output shape: (1, 1, 3, 3)
    ASSERT_EQ(output.getShape(), Shape({1, 1, 3, 3}));

    // Calculate expected output manually
    std::vector<float> expectedOutput = {
        1*1 + 2*0 + 3*(-1) + 2*0 + 3*1 + 4*0 + 3*(-1) + 4*0 + 5*1 + 0.1f,
        2*1 + 3*0 + 4*(-1) + 3*0 + 4*1 + 5*0 + 4*(-1) + 5*0 + 6*1 + 0.1f,
        3*1 + 4*0 + 5*(-1) + 4*0 + 5*1 + 6*0 + 5*(-1) + 6*0 + 7*1 + 0.1f,
        2*1 + 3*0 + 4*(-1) + 3*0 + 4*1 + 5*0 + 4*(-1) + 5*0 + 6*1 + 0.1f,
        3*1 + 4*0 + 5*(-1) + 4*0 + 5*1 + 6*0 + 5*(-1) + 6*0 + 7*1 + 0.1f,
        4*1 + 5*0 + 6*(-1) + 5*0 + 6*1 + 7*0 + 6*(-1) + 7*0 + 8*1 + 0.1f,
        3*1 + 4*0 + 5*(-1) + 4*0 + 5*1 + 6*0 + 5*(-1) + 6*0 + 7*1 + 0.1f,
        4*1 + 5*0 + 6*(-1) + 5*0 + 6*1 + 7*0 + 6*(-1) + 7*0 + 8*1 + 0.1f,
        5*1 + 6*0 + 7*(-1) + 6*0 + 7*1 + 8*0 + 7*(-1) + 8*0 + 9*1 + 0.1f
    };

    // Check the output values
    for (size_t i = 0; i < expectedOutput.size(); ++i) {
        ASSERT_NEAR(output.getData()[i], expectedOutput[i], 1e-5);
    }
}

TEST(Conv2DLayerTest, BackwardPassHardcodedConv2D) {
    Conv2DLayer<float> convLayer(1, 1, 3); // 1 input channel, 1 output channel, 3x3 kernel

    // Set weights and biases
    Tensor<float> weightValues({1, 1, 3, 3}, {1, 0, -1,
                                              0, 1, 0,
                                              -1, 0, 1});
    Tensor<float> biasValues({1, 1}, {0.1f});

    convLayer.setWeights(weightValues);
    convLayer.setBiases(biasValues);

    // Create an input tensor (1 batch, 1 channel, 5x5 image)
    Tensor<float> input({1, 1, 5, 5}, {1, 2, 3, 4, 5,
                                       2, 3, 4, 5, 6,
                                       3, 4, 5, 6, 7,
                                       4, 5, 6, 7, 8,
                                       5, 6, 7, 8, 9});

    // Perform forward pass
    Tensor<float> output = convLayer.forward(input);

    // Create gradient output for backward pass (1 batch, 1 channel, 3x3 output)
    Tensor<float> gradOutput({1, 1, 3, 3}, {1, 1, 1,
                                            1, 1, 1,
                                            1, 1, 1});

    // Perform backward pass
    Tensor<float> gradInput = convLayer.backward(gradOutput);

    // Calculate expected input gradient manually
    std::vector<std::vector<float>> expectedGradInput(5, std::vector<float>(5, 0.0f));
    for (int oh = 0; oh < 3; ++oh) {
        for (int ow = 0; ow < 3; ++ow) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    expectedGradInput[ih][iw] += weightValues.at({0, 0, kh, kw}) * gradOutput.at({0, 0, oh, ow});
                }
            }
        }
    }

    // Check the input gradient values
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            ASSERT_NEAR(gradInput.at({0, 0, i, j}), expectedGradInput[i][j], 1e-5)
                << "Mismatch at position (" << i << ", " << j << ")";
        }
    }

    // Check weight gradients
    Tensor<float> weightGradients = convLayer.getWeightGradients();
    ASSERT_EQ(weightGradients.getShape(), Shape({1, 1, 3, 3}));

    // Calculate expected weight gradients
    std::vector<float> expectedWeightGradients = {
        27, 36, 45,
        36, 45, 54,
        45, 54, 63
    };

    for (size_t i = 0; i < expectedWeightGradients.size(); ++i) {
        ASSERT_NEAR(weightGradients.getData()[i], expectedWeightGradients[i], 1e-5)
            << "Weight gradient mismatch at index " << i;
    }

    // Check bias gradients
    Tensor<float> biasGradients = convLayer.getBiasGradients();
    ASSERT_EQ(biasGradients.getShape(), Shape({1, 1}));
    ASSERT_NEAR(biasGradients.getData()[0], 9.0f, 1e-5); // Sum of all elements in gradOutput
}

TEST(Conv2DLayerTest, DifferentInputShapes) {
    Conv2DLayer<float> convLayer(3, 2, 3); // 3 input channels, 2 output channels, 3x3 kernel

    // Test with single sample (4D input with batch size 1)
    Tensor<float> input1({1, 3, 32, 32});
    Tensor<float> output1 = convLayer.forward(input1);
    ASSERT_EQ(output1.getShape(), Shape({1, 2, 30, 30}));

    // Test with multiple samples (4D input with batch size > 1)
    Tensor<float> input2({5, 3, 32, 32});
    Tensor<float> output2 = convLayer.forward(input2);
    ASSERT_EQ(output2.getShape(), Shape({5, 2, 30, 30}));

    // Perform backward pass with multiple samples
    Tensor<float> gradOutput({5, 2, 30, 30});
    Tensor<float> gradInput = convLayer.backward(gradOutput);
    ASSERT_EQ(gradInput.getShape(), Shape({5, 3, 32, 32}));
}

TEST(Conv2DLayerTest, WeightUpdateWithOptimizer) {
    Conv2DLayer<float> convLayer(1, 1, 3);
    AdamOptions adamOptions;
    adamOptions.learningRate = 0.01f;
    AdamOptimizer<float> optimizer(adamOptions);

    // Perform a forward and backward pass
    Tensor<float> input({1, 1, 5, 5}, 1.0f);  // Initialize with 1s for simplicity
    Tensor<float> output = convLayer.forward(input);
    Tensor<float> gradOutput({1, 1, 3, 3}, 1.0f);  // Initialize with 1s for simplicity
    convLayer.backward(gradOutput);

    // Get initial weights and biases
    Tensor<float> initialWeights = convLayer.getWeights();
    Tensor<float> initialBiases = convLayer.getBiases();

    // Perform weight update
    convLayer.updateWeights(optimizer);

    // Get updated weights and biases
    Tensor<float> updatedWeights = convLayer.getWeights();
    Tensor<float> updatedBiases = convLayer.getBiases();

    // Check that weights and biases have been updated
    ASSERT_FALSE(TensorEquals(initialWeights, updatedWeights));
    ASSERT_FALSE(TensorEquals(initialBiases, updatedBiases));
}

// Checks forward and all gradients against a direct cross-correlation over every batch item and channel.
void ExpectMatchesDirectConvolution(int stride, int padding, int dilation) {
    const int N = 2, C = 3, OC = 4, H = 7, W = 6, KH = 3, KW = 2;
    const int OH = (H + 2 * padding - dilation * (KH - 1) - 1) / stride + 1;
    const int OW = (W + 2 * padding - dilation * (KW - 1) - 1) / stride + 1;
    Conv2DLayer<float> convLayer(C, OC, KH, KW, stride, padding, dilation);

    std::vector<float> x(N * C * H * W), w(OC * C * KH * KW), b(OC), g(N * OC * OH * OW);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * (i % 7) - 0.3f;
    for (size_t i = 0; i < w.size(); ++i) w[i] = 0.05f * (i % 5) - 0.1f;
    for (int o = 0; o < OC; ++o) b[o] = 0.25f * o - 0.5f;
    for (size_t i = 0; i < g.size(); ++i) g[i] = 0.01f * (i % 11) + 0.5f;
    convLayer.setWeights(Tensor<float>({OC, C, KH, KW}, w.data()));
    convLayer.setBiases(Tensor<float>({OC, 1}, b.data()));

    Tensor<float> output = convLayer.forward(Tensor<float>({N, C, H, W}, x.data()));
    ASSERT_EQ(output.getShape(), Shape({N, OC, OH, OW}));
    Tensor<float> gradInput = convLayer.backward(Tensor<float>({N, OC, OH, OW}, g.data()));

    std::vector<float> y(g.size()), dx(x.size()), dw(w.size()), db(OC);
    for (int n = 0; n < N; ++n)
        for (int o = 0; o < OC; ++o)
            for (int oy = 0; oy < OH; ++oy)
                for (int ox = 0; ox < OW; ++ox) {
                    const int yi = ((n * OC + o) * OH + oy) * OW + ox;
                    y[yi] = b[o];
                    db[o] += g[yi];
                    for (int c = 0; c < C; ++c)
                        for (int ky = 0; ky < KH; ++ky)
                            for (int kx = 0; kx < KW; ++kx) {
                                const int iy = oy * stride + ky * dilation - padding, ix = ox * stride + kx * dilation - padding;
                                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                                const int xi = ((n * C + c) * H + iy) * W + ix, wi = ((o * C + c) * KH + ky) * KW + kx;
                                y[yi] += w[wi] * x[xi];
                                dx[xi] += g[yi] * w[wi];
                                dw[wi] += g[yi] * x[xi];
                            }
                }

    auto expectNear = [](const Tensor<float>& actual, const std::vector<float>& expected) {
        ASSERT_EQ(actual.getShape().size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_NEAR(actual.getData()[i], expected[i], 1e-4) << "at flat index " << i;
        }
    };
    expectNear(output, y);
    expectNear(gradInput, dx);
    expectNear(convLayer.getWeightGradients(), dw);
    expectNear(convLayer.getBiasGradients(), db);
}

TEST(Conv2DLayerTest, MatchesDirectConvolutionAcrossBatchAndChannels) {
    ExpectMatchesDirectConvolution(1, 0, 1);
}

TEST(Conv2DLayerTest, MatchesDirectConvolutionWithStridePaddingAndDilation) {
    ExpectMatchesDirectConvolution(2, 1, 2);
}

TEST(Conv2DLayerTest, KeepsDoublePrecision) {
    // In float, 1.00000001 rounds to 1, 1e-100 underflows and 1e100 overflows.
    const double bias = 1e-10;
    Conv2DLayer<double> convLayer(1, 1, 1);
    convLayer.setWeights(Tensor<double>({1, 1, 1, 1}, {1.0}));
    convLayer.setBiases(Tensor<double>({1, 1}, {bias}));
    for (double value : {1.00000001, 1e-100, 1e100}) {
        Tensor<double> output = convLayer.forward(Tensor<double>({1, 1, 1, 1}, {value}));
        convLayer.backward(Tensor<double>({1, 1, 1, 1}, {1.0}));
        EXPECT_EQ(output[0], value + bias);
        EXPECT_EQ(convLayer.getWeightGradients()[0], value);
    }
}

} // namespace smart_dnn

#endif // TEST_CONV_2D_CPP
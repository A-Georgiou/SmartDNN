#ifndef TEST_LOSS_CPP
#define TEST_LOSS_CPP

#include <gtest/gtest.h>
#include "smart_dnn/loss/MSELoss.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"

namespace sdnn {

TEST(MSELossTest, ComputeLossMatchesExpectedValue) {
    MSELoss mseLoss;

    // Prediction and target tensors
    Tensor prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor target({2, 2}, {1.5f, 2.5f, 3.0f, 4.0f});

    // Compute loss
    Tensor loss = mseLoss.compute(prediction, target);

    // Manually compute the expected MSE
    float expectedMSE = ((0.5f * 0.5f) + (0.5f * 0.5f) + (0.0f * 0.0f + 0.0f * 0.0f)) / 4;

    // Check if the computed loss matches the expected value
    EXPECT_NEAR(loss.at<float>(0), expectedMSE, 1e-6)
        << "The computed MSE loss is not correct.";
}

TEST(MSELossTest, ComputeGradientMatchesExpectedValue) {
    MSELoss mseLoss;

    // Prediction and target tensors
    Tensor prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor target({2, 2}, {1.5f, 2.5f, 3.0f, 4.0f});

    // Compute gradient
    Tensor grad = mseLoss.gradient(prediction, target);

    // Manually compute the expected gradient
    std::vector<float> expectedGradient = {-0.25f, -0.25f, 0.0f, 0.0f};

    // Check if the computed gradient matches the expected values
    for (size_t i = 0; i < grad.shape().size(); ++i) {
        EXPECT_NEAR(grad.at<float>(i), expectedGradient[i], 1e-6)
            << "The computed gradient at index " << i << " is not correct.";
    }
}

TEST(MSELossTest, ComputeLossWithBroadcastTarget) {
    MSELoss mseLoss;

    // Prediction is a 2x2 tensor
    Tensor prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    // Target is a 1D tensor with values broadcasted across the rows of prediction
    Tensor target({2}, {1.5f, 3.5f});

    // Compute loss
    Tensor loss = mseLoss.compute(prediction, target);

    // Manually compute the expected MSE with broadcasting
    float expectedMSE = (0.5f * 0.5f + 1.5f * 1.5f + 1.5f * 1.5f + 0.5f * 0.5f) / 4;

    // Check if the computed loss matches the expected value
    EXPECT_NEAR(loss.at<float>(0), expectedMSE, 1e-6)
        << "The computed MSE loss with broadcasting is not correct.";
}

TEST(MSELossTest, GradientWithBroadcastTarget) {
    MSELoss mseLoss;

    // Prediction is a 2x2 tensor
    Tensor prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    // Target is a 1D tensor with values broadcasted across the rows of prediction
    Tensor target({2}, {1.5f, 3.5f});

    // Compute gradient
    Tensor grad = mseLoss.gradient(prediction, target);

    // Manually compute the expected gradient with broadcasting
    std::vector<float> expectedGradient = {-0.25f, -0.75f, 0.75f, 0.25f};

    // Check if the computed gradient matches the expected values
    for (size_t i = 0; i < grad.shape().size(); ++i) {
        EXPECT_NEAR(grad.at<float>(i), expectedGradient[i], 1e-6)
            << "The computed gradient at index " << i << " with broadcasting is not correct.";
    }
}

TEST(MSELossTest, MismatchedShapesThrowsException) {
    MSELoss mseLoss;

    // Mismatched shapes
    Tensor prediction({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor target({3}, {1.0f, 2.0f, 3.0f});

    // Expect an exception to be thrown due to mismatched shapes
    EXPECT_THROW(mseLoss.compute(prediction, target), std::invalid_argument)
        << "Mismatched shapes should throw an exception.";
}

} // namespace sdnn

#endif // TEST_LOSS_CPP
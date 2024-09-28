#ifndef TEST_LOSS_CPP
#define TEST_LOSS_CPP

#include <gtest/gtest.h>
#include "smart_dnn/loss/MSELoss.hpp"
#include "smart_dnn/loss/CategoricalCrossEntropyLoss.hpp"
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
    std::vector<float> expectedGradient = {-0.125f, -0.125f, 0.0f, 0.0f};

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
    std::vector<float> expectedGradient = {-0.125f, -0.375f, 0.375f, 0.125f};

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

TEST(CategoricalCrossEntropyLossTest, ComputeLossMatchesExpectedValue) {
    CategoricalCrossEntropyLoss cceLoss;

    // Prediction and target tensors
    Tensor prediction({2, 2}, {0.1f, 0.9f, 0.8f, 0.2f});
    Tensor target({2, 2}, {0.0f, 1.0f, 1.0f, 0.0f});

    // Compute loss
    Tensor loss = cceLoss.compute(prediction, target);

    // Manually compute the expected loss
    float expectedLoss = -std::log(0.9f) - std::log(0.8f);
    
    EXPECT_NEAR(loss.at<float>(0), expectedLoss/2, 1e-6)
        << "The computed CCE loss is not correct.";
}

TEST(CategoricalCrossEntropyLossTest, ComputeGradientMatchesExpectedValue) {
    CategoricalCrossEntropyLoss cceLoss;

    // Prediction and target tensors
    Tensor prediction({2, 2}, {0.1f, 0.9f, 0.8f, 0.2f});
    Tensor target({2}, {0.0f, 1.0f});

    // Compute gradient
    Tensor grad = cceLoss.gradient(prediction, target);

    std::vector<float> expectedGradient = {-0.45f, 0.45f, 0.4f, -0.4f};

    // Check if the computed gradient matches the expected values
    for (size_t i = 0; i < grad.shape().size(); ++i) { 
        EXPECT_NEAR(grad.at<float>(i), expectedGradient[i], 1e-6)
            << "The computed gradient at index " << i << " is not correct.";
    }
}

TEST(CategoricalCrossEntropyLossTest, ComputeLossWithBroadcastTarget) {
    CategoricalCrossEntropyLoss cceLoss;

    // Prediction is a 2x2 tensor
    Tensor prediction({2, 2}, {0.1f, 0.9f, 0.8f, 0.2f});

    // Target is a 1D tensor with values broadcasted across the rows of prediction
    Tensor target({2}, {0.0f, 1.0f});

    // Compute loss
    Tensor loss = cceLoss.compute(prediction, target);

    // Manually compute the expected loss with broadcasting
    float expectedLoss = (-std::log(0.9f) - std::log(0.2f)) / 2.0f;

    // Check if the computed loss matches the expected value
    EXPECT_NEAR(loss.at<float>(0), expectedLoss, 1e-6)
        << "The computed CCE loss with broadcasting is not correct.";
}


TEST(CategoricalCrossEntropyLossTest, ComputeLossForMNIST) {
    CategoricalCrossEntropyLoss cceLoss;

    Tensor prediction = rand({2, 10});  // 2 samples, 10 classes
    
    prediction = prediction / sum(prediction, {1}, true);

    // 2D target tensor (one-hot encoded)
    Tensor target = zeros({2, 10});
    target.set({0, 3}, 1.0f);  // First sample is class 3
    target.set({1, 7}, 1.0f);  // Second sample is class 7

    // Compute loss
    Tensor loss = cceLoss.compute(prediction, target);

    // Check if the loss is a scalar
    EXPECT_EQ(loss.shape().size(), 1);
    EXPECT_EQ(loss.shape()[0], 1);

    // Check if the loss is finite and non-negative
    EXPECT_TRUE(std::isfinite(loss.at<float>(0))) << "Loss should be finite!";
    EXPECT_GE(loss.at<float>(0), 0) << "Loss should be non-negative!";
}

TEST(CategoricalCrossEntropyLossTest, ComputeGradientForMNIST) {
    CategoricalCrossEntropyLoss cceLoss;

    // 2D prediction tensor (batch_size, num_classes)
    Tensor prediction = rand({2, 10});  // 2 samples, 10 classes
    
    // Normalize predictions to represent probabilities
    prediction = prediction / sum(prediction, {1}, true);

    // 2D target tensor (one-hot encoded)
    Tensor target = zeros({2, 10});
    target.set({0, 3}, 1.0f);  // First sample is class 3
    target.set({1, 7}, 1.0f);  // Second sample is class 7

    // Compute gradient
    Tensor grad = cceLoss.gradient(prediction, target);

    // Check gradient shape
    EXPECT_EQ(grad.shape(), prediction.shape());

    // Ensure that gradients are finite
    for (size_t i = 0; i < grad.shape().size(); ++i) {
        EXPECT_TRUE(std::isfinite(grad.at<float>(i)))
            << "Gradient at index " << i << " should be finite!";
    }
}

TEST(CategoricalCrossEntropyLossTest, ComputeLossForMNISTRobust) {
    float epsilon = 1e-7f;
    CategoricalCrossEntropyLoss cceLoss;

    // Test case 1: Perfect prediction
    {
        Tensor prediction = zeros({1, 10});
        prediction.set({0, 3}, 1.0f);
        Tensor target = zeros({1, 10});
        target.set({0, 3}, 1.0f);

        Tensor loss = cceLoss.compute(prediction, target);
        EXPECT_NEAR(loss.at<float>(0), 0.0f, 1e-6) << "Loss should be close to 0 for perfect prediction";
    }

    // Test case 2: Worst prediction
    {
        Tensor prediction = zeros({1, 10});
        prediction.set({0, 3}, 1.0f);
        Tensor target = zeros({1, 10});
        target.set({0, 7}, 1.0f);

        Tensor loss = cceLoss.compute(prediction, target);
        float expected_loss = -std::log(epsilon);
        EXPECT_NEAR(loss.at<float>(0), expected_loss, 1e-6) << "Loss should be maximum for worst prediction";
    }

    // Test case 3: Realistic prediction
    {
        Tensor prediction = Tensor({1, 3}, {0.3f, 0.5f, 0.2f});
        Tensor target = Tensor({1, 3}, {0.0f, 1.0f, 0.0f});

        Tensor loss = cceLoss.compute(prediction, target);
        float expected_loss = -std::log(0.5f);
        EXPECT_NEAR(loss.at<float>(0), expected_loss, 1e-6) << "Loss should match hand-calculated value";
    }

    // Test case 4: Batch prediction
    {
        Tensor prediction = Tensor({2, 3}, {0.1f, 0.7f, 0.2f, 0.3f, 0.3f, 0.4f});
        Tensor target = Tensor({2, 3}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f});

        Tensor loss = cceLoss.compute(prediction, target);
        float expected_loss = (-std::log(0.7f) - std::log(0.4f)) / 2.0f;
        EXPECT_NEAR(loss.at<float>(0), expected_loss, 1e-6) << "Batch loss should be average of individual losses";
    }
}

TEST(CategoricalCrossEntropyLossTest, ComputeGradientForMNISTRobust) {
    float epsilon = 1e-7f;   
    CategoricalCrossEntropyLoss cceLoss;

    // Test case 1: Perfect prediction
    {
        Tensor prediction = zeros({1, 10});
        prediction.set({0, 3}, 1.0f);
        Tensor target = zeros({1, 10});
        target.set({0, 3}, 1.0f);

        Tensor grad = cceLoss.gradient(prediction, target);
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_NEAR(grad.at<float>({0, i}), 0.0f, 1e-6) << "Gradient should be close to 0 for perfect prediction";
        }
    }

    // Test case 2: Worst prediction
    {
        Tensor prediction = zeros({1, 10});
        prediction.set({0, 3}, 1.0f);
        Tensor target = zeros({1, 10});
        target.set({0, 7}, 1.0f);

        Tensor grad = cceLoss.gradient(prediction, target);

        std::vector<float> expected = {0, 0, 0, 1, 0, 0, 0, -1, 0, 0};
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_NEAR(grad.at<float>({0, i}), expected[i], 1e-6);
        }
    }

    // Test case 3: Realistic prediction
    {
        Tensor prediction = Tensor({1, 3}, {0.3f, 0.5f, 0.2f});
        Tensor target = Tensor({1, 3}, {0.0f, 1.0f, 0.0f});

        Tensor grad = cceLoss.gradient(prediction, target);
        EXPECT_NEAR(grad.at<float>({0, 0}), 0.3f, 1e-6);
        EXPECT_NEAR(grad.at<float>({0, 1}), -0.5f, 1e-6);
        EXPECT_NEAR(grad.at<float>({0, 2}), 0.2f, 1e-6);
    }
}

} // namespace sdnn

#endif // TEST_LOSS_CPP
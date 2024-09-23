#ifndef TEST_OPTIMIZERS_CPP
#define TEST_OPTIMIZERS_CPP

#include <gtest/gtest.h>
#include "tests/utils/tensor_helpers.hpp"
#include "smart_dnn/optimizers/AdamOptimizer.hpp"

namespace sdnn {

TEST(AdamOptimizerTest, OptimizerUpdatesMockWeightsCorrectly) {
    // Mock weight tensor (2x2 matrix initialized with specific values)
    Tensor weights({2, 2}, {0.5f, -0.5f, 0.8f, -0.8f});

    // Mock gradient tensor (2x2 matrix initialized with specific values)
    Tensor gradients({2, 2}, {0.5f, -0.5f, 0.8f, -0.8f});

    // Create an Adam optimizer with default options
    AdamOptions adamOptions;
    adamOptions.learningRate = 0.1f;  // Default learning rate
    AdamOptimizer optimizer(adamOptions);

    // Capture the initial weights for comparison
    Tensor initialWeights = weights.clone();  

    // Apply the Adam optimizer directly to update the mock weights
    std::vector<std::reference_wrapper<Tensor>> weightList = {weights};
    std::vector<std::reference_wrapper<Tensor>> gradientList = {gradients};
    optimizer.optimize(weightList, gradientList);

    // Capture the updated weights
    Tensor updatedWeights = weights;

    // Verify that weights have been updated (they should be different after optimization)
    for (size_t i = 0; i < updatedWeights.shape().size(); ++i) {
        EXPECT_NE(updatedWeights.at<float>(i), initialWeights.at<float>(i))
            << "Weight at index " << i << " was not updated!";
    }
}

TEST(AdamOptimizerTest, MultipleOptimizationSteps) {
    // Mock weight tensor
    Tensor weights({2, 2}, {0.5f, -0.5f, 0.8f, -0.8f});

    // Mock gradient tensor
    Tensor gradients({2, 2}, {0.1f, -0.1f, 0.2f, -0.2f});

    // Create an Adam optimizer with default options
    AdamOptions adamOptions;
    adamOptions.learningRate = 0.1f;
    AdamOptimizer optimizer(adamOptions);

    // Apply the optimizer multiple times and verify weights change after each step
    std::vector<std::reference_wrapper<Tensor>> weightList = {weights};
    std::vector<std::reference_wrapper<Tensor>> gradientList = {gradients};

    for (int i = 0; i < 5; ++i) {
        Tensor prevWeights = weights.clone();
        optimizer.optimize(weightList, gradientList);
        for (size_t j = 0; j < weights.shape().size(); ++j) {
            EXPECT_NE(weights.at<float>(j), prevWeights.at<float>(j))
                << "Weight at index " << j << " did not change after optimization step " << i + 1;
        }
    }
}

TEST(AdamOptimizerTest, DifferentLearningRates) {
    Tensor weights({2, 2}, {0.5f, -0.5f, 0.8f, -0.8f});
    Tensor gradients({2, 2}, {0.1f, -0.1f, 0.2f, -0.2f});

    // Adam optimizer with small learning rate
    AdamOptions smallLR;
    smallLR.learningRate = 0.001f;
    AdamOptimizer optimizerSmallLR(smallLR);

    // Adam optimizer with large learning rate
    AdamOptions largeLR;
    largeLR.learningRate = 0.5f;
    AdamOptimizer optimizerLargeLR(largeLR);

    Tensor weightsSmallLR = weights.clone();
    Tensor weightsLargeLR = weights.clone();

    optimizerSmallLR.optimize({weightsSmallLR}, {gradients});
    optimizerLargeLR.optimize({weightsLargeLR}, {gradients});

    // Verify that the large learning rate results in a bigger weight update
    for (size_t i = 0; i < weights.shape().size(); ++i) {
        float diffSmall = std::abs(weights.at<float>(i) - weightsSmallLR.at<float>(i));
        float diffLarge = std::abs(weights.at<float>(i) - weightsLargeLR.at<float>(i));
        EXPECT_GT(diffLarge, diffSmall)
            << "Weight update with large learning rate should be greater than with small learning rate";
    }
}

TEST(AdamOptimizerTest, ZeroGradientsNoWeightChange) {
    Tensor weights({2, 2}, {0.5f, -0.5f, 0.8f, -0.8f});
    Tensor zeroGradients({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});

    // Create an Adam optimizer with default options
    AdamOptions adamOptions;
    adamOptions.learningRate = 0.1f;
    AdamOptimizer optimizer(adamOptions);

    Tensor initialWeights = weights.clone();

    // Apply the optimizer with zero gradients
    optimizer.optimize({weights}, {zeroGradients});

    // Verify that the weights have not changed
    for (size_t i = 0; i < weights.shape().size(); ++i) {
        EXPECT_EQ(weights.at<float>(i), initialWeights.at<float>(i))
            << "Weights should not change with zero gradients";
    }
}

TEST(AdamOptimizerTest, NonDefaultBetaValues) {
    Tensor weights({2, 2}, {0.5f, -0.5f, 0.8f, -0.8f});
    Tensor gradients({2, 2}, {0.1f, -0.1f, 0.2f, -0.2f});

    // Adam optimizer with default beta values
    AdamOptions defaultBeta;
    defaultBeta.learningRate = 0.1f;
    AdamOptimizer optimizerDefaultBeta(defaultBeta);

    // Adam optimizer with modified beta values
    AdamOptions customBeta;
    customBeta.learningRate = 0.1f;
    customBeta.beta1 = 0.5f;
    customBeta.beta2 = 0.99f;
    AdamOptimizer optimizerCustomBeta(customBeta);

    Tensor weightsDefaultBeta = weights.clone();
    Tensor weightsCustomBeta = weights.clone();

    optimizerDefaultBeta.optimize({weightsDefaultBeta}, {gradients});
    optimizerCustomBeta.optimize({weightsCustomBeta}, {gradients});

    // Verify that the weight updates are different with custom beta values
    for (size_t i = 0; i < weights.shape().size(); ++i) {
        EXPECT_NE(weightsDefaultBeta.at<float>(i), weightsCustomBeta.at<float>(i))
            << "Weight updates with default and custom beta values should differ";
    }
}

}   // namespace sdnn

#endif // TEST_OPTIMIZERS_CPP
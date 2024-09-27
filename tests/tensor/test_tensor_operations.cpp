#include <gtest/gtest.h>
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/Backend/Default/AdvancedTensorOperations.hpp"

namespace sdnn {

class TensorOperationsTest : public ::testing::Test {
protected:
    Tensor createTensor(const std::vector<int>& shape_, const std::vector<float>& values) {
        return Tensor(Shape(shape_), values);
    }
};


TEST_F(TensorOperationsTest, ReciprocalTest) {
    Tensor a = createTensor({2, 2}, {0.5f, 0.25f, 0.125f, 0.0625f});
    Tensor b = createTensor({2, 2}, {1.0f, 2.0f, 4.0f, 8.0f});

    Tensor result = AdvancedTensorOperations::reciprocal(a);
    Tensor result2 = AdvancedTensorOperations::reciprocal(b);

    std::vector<float> expected = {2.0f, 4.0f, 8.0f, 16.0f};
    std::vector<float> expected2 = {1.0f, 0.5f, 0.25f, 0.125f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
        EXPECT_NEAR(result2.at<float>(i), expected2[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, TransposeTest) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f,
                                     3.0f, 4.0f});
    Tensor b = createTensor({2, 2, 2}, {1.0f, 2.0f,
                                        3.0f, 4.0f,
                                        5.0f, 6.0f,
                                        7.0f, 8.0f});

    Tensor result = AdvancedTensorOperations::transpose(a, 1, 0);
    Tensor result2 = AdvancedTensorOperations::transpose(b, 1, 0);

    std::vector<float> expected = {1.0f, 3.0f, 2.0f, 4.0f};
    std::vector<float> expected2 = {1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
        EXPECT_NEAR(result2.at<float>(i), expected2[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, VarianceTest) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f,
                                     3.0f, 4.0f});
    Tensor mean_out = mean(a, {0}, false); // Should be [2.0, 3.0]
    Tensor result = AdvancedTensorOperations::variance(a, mean_out, {0});
    
    std::vector<float> expected = {1.0f, 1.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, MeanFunctionTest) {
    // Create a 2D tensor
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f});

    // Compute mean over axis 0, keepDims=false
    Tensor mean_axis0 = mean(a, {0}, false);
    EXPECT_EQ(mean_axis0.shape(), Shape({3}));
    std::vector<float> expected_mean_axis0 = {2.5f, 3.5f, 4.5f}; // (1+4)/2, (2+5)/2, (3+6)/2
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(mean_axis0.at<float>(i), expected_mean_axis0[i], 1e-5);
    }

    // Compute mean over axis 0, keepDims=true
    Tensor mean_axis0_keep = mean(a, {0}, true);
    EXPECT_EQ(mean_axis0_keep.shape(), Shape({1, 3}));
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(mean_axis0_keep.at<float>(i), expected_mean_axis0[i], 1e-5);
    }

    // Compute mean over axis 1, keepDims=false
    Tensor mean_axis1 = mean(a, {1}, false);
    EXPECT_EQ(mean_axis1.shape(), Shape({2}));
    std::vector<float> expected_mean_axis1 = {2.0f, 5.0f}; // (1+2+3)/3, (4+5+6)/3
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(mean_axis1.at<float>(i), expected_mean_axis1[i], 1e-5);
    }

    // Compute mean over axis 1, keepDims=true
    Tensor mean_axis1_keep = mean(a, {1}, true);
    EXPECT_EQ(mean_axis1_keep.shape(), Shape({2, 1}));
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(mean_axis1_keep.at<float>(i), expected_mean_axis1[i], 1e-5);
    }

    // Compute mean over all axes, keepDims=false
    Tensor mean_all = mean(a, {0, 1}, false);
    EXPECT_EQ(mean_all.shape().rank(), 1); // Scalar tensor
    float expected_mean_all = 3.5f; // (1+2+3+4+5+6)/6
    EXPECT_NEAR(mean_all.at<float>(0), expected_mean_all, 1e-5);

    // Compute mean over all axes, keepDims=true
    Tensor mean_all_keep = mean(a, {0, 1}, true);
    EXPECT_EQ(mean_all_keep.shape(), Shape({1, 1}));
    EXPECT_NEAR(mean_all_keep.at<float>(0), expected_mean_all, 1e-5);
}

TEST_F(TensorOperationsTest, VarianceFunctionTest) {
    // Create a 2D tensor
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f});

    // Compute mean over axis 0
    Tensor mean_axis0 = mean(a, {0}, false); // Shape: (3)

    // Compute variance over axis 0
    Tensor var_axis0 = AdvancedTensorOperations::variance(a, mean_axis0, {0});
    EXPECT_EQ(var_axis0.shape(), Shape({3}));

    std::vector<float> expected_var_axis0 = {2.25f, 2.25f, 2.25f}; // Variance calculations
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(var_axis0.at<float>(i), expected_var_axis0[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, MeanFunctionShapeTest) {
    // Test to verify that the output shape matches expectations

    // Create a 3D tensor
    Tensor a = createTensor({2, 3, 4}, std::vector<float>(24, 1.0f)); // All ones

    // Compute mean over axis 0, keepDims=false
    Tensor mean_axis0 = mean(a, {0}, false);
    EXPECT_EQ(mean_axis0.shape(), Shape({3, 4}));

    // Compute mean over axis 0, keepDims=true
    Tensor mean_axis0_keep = mean(a, {0}, true);
    EXPECT_EQ(mean_axis0_keep.shape(), Shape({1, 3, 4}));

    // Compute mean over axes 0 and 2, keepDims=false
    Tensor mean_axes = mean(a, {0, 2}, false);
    EXPECT_EQ(mean_axes.shape(), Shape({3}));

    // Compute mean over axes 0 and 2, keepDims=true
    Tensor mean_axes_keep = mean(a, {0, 2}, true);
    EXPECT_EQ(mean_axes_keep.shape(), Shape({1, 3, 1}));
}

}
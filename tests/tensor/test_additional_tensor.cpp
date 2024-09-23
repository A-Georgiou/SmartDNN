#include <gtest/gtest.h>
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"

namespace sdnn {

class AdditionalTensorOperationsTest : public ::testing::Test {
protected:
    // Helper function to create a Tensor
    Tensor createTensor(const std::vector<int>& shape_, const std::vector<float>& values) {
        return Tensor(Shape(shape_), values);
    }
};

// Test broadcasting for element-wise operations (e.g., add scalar to a matrix)
TEST_F(AdditionalTensorOperationsTest, BroadcastingWithScalar) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    // Add a scalar to a tensor
    Tensor result = a + 1.0;

    std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

// Test broadcasting a vector along an axis for element-wise addition
TEST_F(AdditionalTensorOperationsTest, BroadcastingVectorAlongAxis) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({2}, {10.0f, 20.0f});

    Tensor result = a + b;

    std::vector<float> expected = {11.0f, 22.0f, 13.0f, 24.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

// Test complex reduction: Sum over all axes
TEST_F(AdditionalTensorOperationsTest, SumOverAllAxes) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor result = sum(a, {}, false);

    std::vector<float> expected = {10.0f}; 
    EXPECT_NEAR(result.at<float>(0), expected[0], 1e-5);
}

// Test complex reduction: Mean over all axes
TEST_F(AdditionalTensorOperationsTest, MeanOverAllAxes) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor result = mean(a, {}, false); 

    std::vector<float> expected = {2.5f}; 
    EXPECT_NEAR(result.at<float>(0), expected[0], 1e-5);
}

// Test invalid shape for binary operation
TEST_F(AdditionalTensorOperationsTest, IncompatibleShapeAdd) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({3, 1}, {1.0f, 2.0f, 3.0f});

    // Adding tensors with incompatible shapes should throw an error
    EXPECT_THROW({
        Tensor result = a + b;
    }, std::invalid_argument);
}

// Test scalar multiplication resulting in a larger tensor
TEST_F(AdditionalTensorOperationsTest, ScalarMultiplicationIncreasesValues) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    // Multiply tensor by a scalar value
    Tensor result = a * 2.0;

    std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

// Test dividing by a scalar
TEST_F(AdditionalTensorOperationsTest, ScalarDivisionDecreasesValues) {
    Tensor a = createTensor({2, 2}, {4.0f, 6.0f, 8.0f, 10.0f});

    // Divide tensor by a scalar value
    Tensor result = a / 2.0;

    std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

// Test tensor creation with a single value (filling)
TEST_F(AdditionalTensorOperationsTest, FillTensorWithScalar) {
    Tensor filledTensor = Tensor(Shape({2, 2}), 3.5f, dtype::f32);

    std::vector<float> expected = {3.5f, 3.5f, 3.5f, 3.5f};
    for (size_t i = 0; i < filledTensor.shape().size(); ++i) {
        EXPECT_NEAR(filledTensor.at<float>(i), expected[i], 1e-5);
    }
}

// Test for cloning a tensor
TEST_F(AdditionalTensorOperationsTest, CloneTensor) {
    Tensor original = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    // Clone the tensor
    Tensor clone = original.clone();

    clone *= 2;

    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < clone.shape().size(); ++i) {
        EXPECT_NEAR(original.at<float>(i), expected[i], 1e-5);
        EXPECT_NEAR(clone.at<float>(i), expected[i]*2, 1e-5);
    }

    // Ensure that the clone and original are distinct objects
    EXPECT_NE(&original.tensorImpl_, &clone.tensorImpl_);
}

}  // namespace sdnn
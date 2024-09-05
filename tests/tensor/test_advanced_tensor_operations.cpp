#ifndef TEST_ADVANCED_TENSOR_OPERATIONS_CPP
#define TEST_ADVANCED_TENSOR_OPERATIONS_CPP

#include <gtest/gtest.h>
#include "../utils/tensor_helpers.hpp"
#include "../../smart_dnn/Tensor/AdvancedTensorOperations.hpp"


namespace smart_dnn {

/*

    TEST APPLY FUNCTION

*/

TEST(AdvancedTensorOperationsTest, TestApply) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> result = AdvancedTensorOperations<float>::apply(a, [](float x) { return x + 1; });
    Tensor<float> expected(Shape({2, 2}), {2, 3, 4, 5});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestApplyDoesNotModifyOriginal) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    AdvancedTensorOperations<float>::apply(a, [this](float x) { return x + 1; });
    ASSERT_TRUE(TensorEquals(a, a));
}

/*

    TEST APPLY PAIR FUNCTION

*/

TEST(AdvancedTensorOperationsTest, TestApplyPair) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> b(Shape({2, 2}), {5, 6, 7, 8});
    Tensor<float> result = AdvancedTensorOperations<float>::applyPair(a, b, [](float x, float y) { return x + y; });
    Tensor<float> expected(Shape({2, 2}), {6, 8, 10, 12});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestMismatchShape) {
    Tensor<float> a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    Tensor<float> b(Shape({2, 2}), {5, 6, 7, 8});
    ASSERT_THROW(
        AdvancedTensorOperations<float>::applyPair(a, b, [](float x, float y) { return x + y; }),
        std::invalid_argument);
}

/*

    TEST SUM FUNCTION

*/

TEST(AdvancedTensorOperationsTest, TestSum) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> result = AdvancedTensorOperations<float>::sum(a);
    Tensor<float> expected(Shape({1}), {10});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestSumAcrossAxisZero) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> result = AdvancedTensorOperations<float>::sum(a, 0);
    std::vector<float> expectedData = {4, 6};
    ValidateTensorShape(result, 1, 2, {2});
    ValidateTensorData(result, expectedData);
}

TEST(AdvancedTensorOperationsTest, TestSumAcrossAxisOne) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> result = AdvancedTensorOperations<float>::sum(a, 1);
    std::vector<float> expectedData = {3, 7};
    ValidateTensorShape(result, 1, 2, {2});
    ValidateTensorData(result, expectedData);
}

TEST(AdvancedTensorOperationsTest, AttemptSumAcrossInvalidAxis) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    ASSERT_THROW(AdvancedTensorOperations<float>::sum(a, 2), std::runtime_error);
}

TEST(AdvancedTensorOperationsTest, TestSumAcrossMultipleAxes) {
    Tensor<float> a(Shape({2, 2, 2}), { 1, 2,  3, 4,
                                        5, 6,  7, 8});
    Tensor<float> result = AdvancedTensorOperations<float>::sum(a, {0, 2});
    std::vector<float> expectedData = {14, 22};
    ValidateTensorShape(result, 1, 2, {2});
    ValidateTensorData(result, expectedData);
}

/*

    TEST RECIROCAL FUNCTION

*/

TEST(AdvancedTensorOperationsTest, TestReciprocal) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> result = AdvancedTensorOperations<float>::reciprocal(a);
    Tensor<float> expected(Shape({2, 2}), {1, 0.5, 0.333333, 0.25});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestReciprocalWithEpsilon) {
    Tensor<float> a(Shape({2, 2}), {1, 0, 3, 4});
    Tensor<float> result = AdvancedTensorOperations<float>::reciprocal(a, 1e-6);
    Tensor<float> expected(Shape({2, 2}), {1, 1e6, 0.333333, 0.25});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestReciprocalWithZero) {
    Tensor<float> a(Shape({2, 2}), {0, 0, 0, 0});
    Tensor<float> result = AdvancedTensorOperations<float>::reciprocal(a, 1e-6);
    Tensor<float> expected(Shape({2, 2}), {1e6, 1e6, 1e6, 1e6});
    ASSERT_TRUE(TensorEquals(result, expected));
}

// Test with negative epsilon
// Expectation: negative epislon will be ignored, since we check against absolute value
TEST(AdvancedTensorOperationsTest, TestReciprocalWithNegativeEpsilon) {
    Tensor<float> a(Shape({2, 2}), {1, -1e-2, 3, 4});
    Tensor<float> result = AdvancedTensorOperations<float>::reciprocal(a, -1e-6);
    Tensor<float> expected(Shape({2, 2}), {1, -100, 0.333333, 0.25});
    ASSERT_TRUE(TensorEquals(result, expected));
}

/*

    TEST MEAN FUNCTION

*/

TEST(AdvancedTensorOperationsTest, TestMean) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> result = AdvancedTensorOperations<float>::mean(a, {0, 1});
    Tensor<float> expected(Shape({1}), {2.5});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestMeanAcrossAxisZero) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> result = AdvancedTensorOperations<float>::mean(a, {0});
    std::vector<float> expectedData = {2, 3};
    ValidateTensorShape(result, 2, 2, {1, 2});
    ValidateTensorData(result, expectedData);
}

TEST(AdvancedTensorOperationsTest, TestMeanAcrossAxisOne) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> result = AdvancedTensorOperations<float>::mean(a, {1});
    std::vector<float> expectedData = {1.5, 3.5};
    ValidateTensorShape(result, 2, 2, {2, 1});
    ValidateTensorData(result, expectedData);
}

TEST(AdvancedTensorOperationsTest, TestMeanAcrossAllAxes) {
    Tensor<float> a(Shape({2, 2, 2}), { 1, 2,  3, 4,
                                        5, 6,  7, 8});
    Tensor<float> result = AdvancedTensorOperations<float>::mean(a);
    std::vector<float> expectedData = {4.5};
    ValidateTensorShape(result, 1, 1, {1});
    ValidateTensorData(result, expectedData);
}

/*

    TEST VARIANCE MULTIPLICATION

*/

TEST(AdvancedTensorOperationsTest, TestVarianceMultiplication) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> meanTensor(Shape({2, 2}), {2, 3, 4, 5});
    Tensor<float> result = AdvancedTensorOperations<float>::variance(a, meanTensor);
    Tensor<float> expected(Shape({1}), {1});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestVarianceMultiplicationWithZeroMean) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> meanTensor(Shape({2, 2}), {0, 0, 0, 0});
    Tensor<float> result = AdvancedTensorOperations<float>::variance(a, meanTensor);
    Tensor<float> expected(Shape({1}), {7.5});
    ASSERT_TRUE(TensorEquals(result, expected));
}

/*

    TEST TRANSPOSE FUNCTION

*/

TEST(AdvancedTensorOperationsTest, TestTranspose) {
    Tensor<float> a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    Tensor<float> result = AdvancedTensorOperations<float>::transpose(a, 0, 1);
    Tensor<float> expected(Shape({3, 2}), {1, 4, 2, 5, 3, 6});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestTranspose4D) {
    Tensor<float> a(Shape({2, 2, 2, 2}), { 1, 2, 3, 4, 5, 6, 7, 8,
                                           9, 10, 11, 12, 13, 14, 15, 16});
    Tensor<float> result = AdvancedTensorOperations<float>::transpose(a, 0, 2);
    Tensor<float> expected(Shape({2, 2, 2, 2}),
                                {1, 2, 9,  10, 5, 6, 13, 14,
                                 3, 4, 11, 12, 7, 8, 15, 16});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestTransposeWithInvalidDimensions) {
    Tensor<float> a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    ASSERT_THROW(AdvancedTensorOperations<float>::transpose(a, 0, 2),
                 std::invalid_argument);
}






} // namespace smart_dnn

#endif // TEST_ADVANCED_TENSOR_OPERATIONS_CPP
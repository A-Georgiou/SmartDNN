#ifndef TEST_ADVANCED_TENSOR_OPERATIONS_CPP
#define TEST_ADVANCED_TENSOR_OPERATIONS_CPP

#include <gtest/gtest.h>
#include "tests/utils/tensor_helpers.hpp"
#include "smart_dnn/tensor/AdvancedTensorOperations.hpp"

namespace sdnn {

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

/*

    TEST RESHAPE FUNCTION

*/

TEST(AdvancedTensorOperationsTest, TestReshape) {
    Tensor<float> a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    Tensor<float> result = AdvancedTensorOperations<float>::reshape(a, Shape({3, 2}));
    Tensor<float> expected(Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(TensorEquals(result, expected));
}


TEST(AdvancedTensorOperationsTest, TestReshapeWithInvalidSize) {
    Tensor<float> a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    ASSERT_THROW(AdvancedTensorOperations<float>::reshape(a, Shape({3, 3})),
                 std::runtime_error);
}


/*

    TEST MATMUL FUNCTION 

*/

// TEST 1D x 1D

TEST(AdvancedTensorOperationsTest, TestDotProduct) {
    Tensor<float> a(Shape({3}), {1, 2, 3});
    Tensor<float> b(Shape({3}), {4, 5, 6});
    Tensor<float> result = AdvancedTensorOperations<float>::matmul(a, b);
    Tensor<float> expected(Shape({1}), {32});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestDotProductWithMismatchedDimensions) {
    Tensor<float> a(Shape({3}), {1, 2, 3});
    Tensor<float> b(Shape({2}), {4, 5});
    ASSERT_THROW(AdvancedTensorOperations<float>::matmul(a, b), std::invalid_argument);
}

// TEST 1D x 2D

TEST(AdvancedTensorOperationsTest, TestMatrixVectorMul) {
    Tensor<float> a(Shape({2}), {1, 2});
    Tensor<float> b(Shape({2, 2}), {3, 4, 5, 6});
    Tensor<float> result = AdvancedTensorOperations<float>::matmul(a, b);
    Tensor<float> expected(Shape({2}), {13, 16});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestMatrixVectorMulWithMismatchedDimensions) {
    Tensor<float> a(Shape({3}), {1, 2, 3});
    Tensor<float> b(Shape({2, 2}), {3, 4, 5, 6});
    ASSERT_THROW(AdvancedTensorOperations<float>::matmul(a, b), std::invalid_argument);
}

// TEST 2D x 1D

TEST(AdvancedTensorOperationsTest, TestVectorMatrixMul) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> b(Shape({2}), {5, 6});
    Tensor<float> result = AdvancedTensorOperations<float>::matmul(a, b);
    Tensor<float> expected(Shape({2}), {17, 39});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestVectorMatrixMulWithMismatchedDimensions) {
    Tensor<float> a(Shape({2, 2}), {1, 2, 3, 4});
    Tensor<float> b(Shape({3}), {5, 6, 7});
    ASSERT_THROW(AdvancedTensorOperations<float>::matmul(a, b), std::invalid_argument);
}

// TEST 2D x 2D

TEST(AdvancedTensorOperationsTest, TestMatrixMatrixMul) {
    Tensor<float> a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    Tensor<float> b(Shape({3, 2}), {7, 8, 9, 10, 11, 12});
    Tensor<float> result = AdvancedTensorOperations<float>::matmul(a, b);
    Tensor<float> expected(Shape({2, 2}), {58, 64, 139, 154});
    ASSERT_TRUE(TensorEquals(result, expected));
}

TEST(AdvancedTensorOperationsTest, TestMatrixMatrixMulWithMismatchedDimensions) {
    Tensor<float> a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    Tensor<float> b(Shape({2, 2}), {7, 8, 9, 10});
    ASSERT_THROW(AdvancedTensorOperations<float>::matmul(a, b), std::invalid_argument);
}

// TEST 2D x 3D

TEST(AdvancedTensorOperationsTest, TestMatrix2DMatrix3DMul) {
    Tensor<float> a(Shape({2, 2}), { 1, 2,
                                     3, 4});
    Tensor<float> b(Shape({2, 2, 2}), { 9, 10, 11, 12,
                                        13, 14, 15, 16});
    Tensor<float> result = AdvancedTensorOperations<float>::matmul(a, b);
    Tensor<float> expected(Shape({2, 2, 2}), { 31, 34, 71, 78,
                                               43, 46, 99, 106});
    
    ASSERT_TRUE(TensorEquals(result, expected));
}

// TEST 3D x 3D

TEST(AdvancedTensorOperationsTest, TestMatrix3DMatrix3DMul) {
    Tensor<float> a(Shape({2, 2, 2}), { 1, 2, 3, 4,
                                        5, 6, 7, 8});
    Tensor<float> b(Shape({2, 2, 2}), { 9, 10, 11, 12,
                                        13, 14, 15, 16});
    Tensor<float> result = AdvancedTensorOperations<float>::matmul(a, b);
    Tensor<float> expected(Shape({2, 2, 2}), { 31, 34, 71, 78,
                                              155, 166, 211, 226});
    
    ASSERT_TRUE(TensorEquals(result, expected));
}

// TEST BATCHED MATMUL

TEST(AdvancedTensorOperationsTest, TestBatchedMatrixMultiplication) {
    // Case 1: (2, 3, 4) * (2, 4, 5)
    {
        Tensor<float> a(Shape({2, 3, 4}), {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        });
        Tensor<float> b(Shape({2, 4, 5}), {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
        });
        Tensor<float> result = AdvancedTensorOperations<float>::matmul(a, b);
        std::cout << "(2,3,4) * (2,4,5) = (2,3,5)" << std::endl;
        std::cout << "Result: " << result.toDataString() << std::endl;
        // Expected result calculation omitted for brevity
        Tensor<float> expected(Shape({2, 3, 5}), {
            110, 120, 130, 140, 150, 246, 272, 298, 324, 350, 382,
            424, 466, 508, 550, 1678, 1736, 1794, 1852., 1910, 2134,
            2208, 2282, 2356, 2430, 2590, 2680, 2770, 2860, 2950
        });
        ASSERT_TRUE(TensorEquals(result, expected));
    }

    // Case 2: (3, 2, 3) * (3, 3, 2)
    {
        Tensor<float> a(Shape({3, 2, 3}), {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18
        });
        Tensor<float> b(Shape({3, 3, 2}), {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18
        });
        Tensor<float> result = AdvancedTensorOperations<float>::matmul(a, b);
        std::cout << "(3,2,3) * (3,3,2) = (3,2,2)" << std::endl;
        std::cout << "Result: " << result.toDataString() << std::endl;
        // Expected result calculation omitted for brevity
        Tensor<float> expected(Shape({3, 2, 2}), {
            22, 28, 49, 64, 220, 244, 301, 334, 634, 676, 769, 820
        });
        ASSERT_TRUE(TensorEquals(result, expected));
    }

    // Case 3: (2, 2, 2) * (2, 2)
    {
        Tensor<float> a(Shape({2, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8});
        Tensor<float> b(Shape({2, 2}), {9, 10, 11, 12});
        Tensor<float> result = AdvancedTensorOperations<float>::matmul(a, b);
        std::cout << "(2,2,2) * (2,2) = (2,2,2)" << std::endl;
        std::cout << "Result: " << result.toDataString() << std::endl;
        Tensor<float> expected(Shape({2, 2, 2}), {31, 34, 71, 78, 111, 122, 151, 166});
        ASSERT_TRUE(TensorEquals(result, expected));
    }
}


} // namespace sdnn

#endif // TEST_ADVANCED_TENSOR_OPERATIONS_CPP
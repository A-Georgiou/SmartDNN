#include <gtest/gtest.h>
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"

namespace sdnn {

class TensorAdditionalOperationsTest : public ::testing::Test {
protected:
    Tensor createTensor(const std::vector<int>& shape_, const std::vector<float>& values) {
        return Tensor(Shape(shape_), values);
    }
};

TEST_F(TensorAdditionalOperationsTest, MatmulTwoTensors) {
    // Tensor a: (2x3)
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Tensor b: (3x2)
    Tensor b = createTensor({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    // Expected result of matmul(a, b) should be (2x2)
    Tensor result = matmul(a, b);
    std::vector<float> expected = {
        58.0f, 64.0f,  // First row result
        139.0f, 154.0f // Second row result
    };

    EXPECT_EQ(result.shape(), Shape({2, 2}));
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, TransposeTensor) {
    // Tensor a: (2x3)
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Expected result after transpose (3x2)
    Tensor result = transpose(a, {1, 0});
    std::vector<float> expected = {
        1.0f, 4.0f, // First column becomes first row
        2.0f, 5.0f, // Second column becomes second row
        3.0f, 6.0f  // Third column becomes third row
    };

    EXPECT_EQ(result.shape(), Shape({3, 2}));
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, ReshapeTensor) {
    // Tensor a: (2x3)
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Reshape to (3x2)
    Tensor result = reshape(a, {3, 2});
    std::vector<float> expected = {
        1.0f, 2.0f, 
        3.0f, 4.0f, 
        5.0f, 6.0f
    };

    EXPECT_EQ(result.shape(), Shape({3, 2}));
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }

    // Reshape to 1D tensor (6)
    Tensor result1D = reshape(a, {6});
    std::vector<float> expected1D = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    EXPECT_EQ(result1D.shape(), Shape({6}));
    for (size_t i = 0; i < result1D.shape().size(); ++i) {
        EXPECT_NEAR(result1D.at<float>(i), expected1D[i], 1e-5);
    }
}

/*

    TEST CLONE FUNCTION

*/

TEST_F(TensorAdditionalOperationsTest, CloneFunctionDoesNotModifyOriginal) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor result = a.clone();
    a += 1;
    ASSERT_TRUE(result != a); 
}

/*

    TEST APPLY PAIR FUNCTION

*/

TEST_F(TensorAdditionalOperationsTest, ApplyPairFunction) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

    Tensor result = a + b;  // This should apply an element-wise addition.

    Tensor expected = createTensor({2, 2}, {6.0f, 8.0f, 10.0f, 12.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, ApplyPairFunctionWithMismatchShape) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b = createTensor({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

    ASSERT_THROW(a + b, std::invalid_argument);  // Tensor addition requires matching shapes.
}

/*

    TEST SUM FUNCTION

*/

TEST_F(TensorAdditionalOperationsTest, SumFunction) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor result = sum(a);

    Tensor expected = createTensor({1}, {10.0f});
    EXPECT_NEAR(result.at<float>(0), expected.at<float>(0), 1e-5);
}

TEST_F(TensorAdditionalOperationsTest, SumAcrossAxisZero) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor result = sum(a, {0});

    std::vector<float> expectedData = {4.0f, 6.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expectedData[i], 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, SumAcrossAxisOne) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor result = sum(a, {1});

    std::vector<float> expectedData = {3.0f, 7.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expectedData[i], 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, AttemptSumAcrossInvalidAxis) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ASSERT_THROW(sum(a, {2}), std::invalid_argument);
}

/*

    TEST TRANSPOSE FUNCTION

*/

TEST_F(TensorAdditionalOperationsTest, TransposeFunction) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor result = transpose(a, {1, 0});

    Tensor expected = createTensor({3, 2}, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, TransposeWithInvalidDimensions) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    ASSERT_THROW(transpose(a, {0, 2}), std::invalid_argument);
}

/*

    TEST RESHAPE FUNCTION

*/

TEST_F(TensorAdditionalOperationsTest, ReshapeFunction) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor result = reshape(a, {3, 2});

    Tensor expected = createTensor({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, ReshapeWithInvalidSize) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    ASSERT_THROW(reshape(a, {3, 3}), std::runtime_error);
}

/*

    TEST MATMUL FUNCTION 

*/

// TEST 1D x 1D

TEST_F(TensorAdditionalOperationsTest, Matmul1D1D) {
    Tensor a = createTensor({3}, {1.0f, 2.0f, 3.0f});
    Tensor b = createTensor({3}, {4.0f, 5.0f, 6.0f});
    Tensor result = matmul(a, b);

    Tensor expected = createTensor({1}, {32.0f});
    EXPECT_NEAR(result.at<float>(0), expected.at<float>(0), 1e-5);
}

TEST_F(TensorAdditionalOperationsTest, Matmul1D1DMismatchedDimensions) {
    Tensor a = createTensor({3}, {1.0f, 2.0f, 3.0f});
    Tensor b = createTensor({2}, {4.0f, 5.0f});
    ASSERT_THROW(matmul(a, b), std::invalid_argument);
}

// TEST 1D x 2D

TEST_F(TensorAdditionalOperationsTest, Matmul1D2D) {
    Tensor a = createTensor({2}, {1.0f, 2.0f});
    Tensor b = createTensor({2, 2}, {3.0f, 4.0f, 5.0f, 6.0f});
    Tensor result = matmul(a, b);

    Tensor expected = createTensor({2}, {13.0f, 16.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, Matmul1D2DMismatchedDimensions) {
    Tensor a = createTensor({3}, {1.0f, 2.0f, 3.0f});
    Tensor b = createTensor({2, 2}, {3.0f, 4.0f, 5.0f, 6.0f});
    ASSERT_THROW(matmul(a, b), std::invalid_argument);
}

// TEST 2D x 1D

TEST_F(TensorAdditionalOperationsTest, Matmul2D1D) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({2}, {5.0f, 6.0f});
    Tensor result = matmul(a, b);

    Tensor expected = createTensor({2}, {17.0f, 39.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, Matmul2D1DMismatchedDimensions) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({3}, {5.0f, 6.0f, 7.0f});
    ASSERT_THROW(matmul(a, b), std::invalid_argument);
}

// TEST 2D x 2D

TEST_F(TensorAdditionalOperationsTest, Matmul2D2D) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b = createTensor({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor result = matmul(a, b);

    Tensor expected = createTensor({2, 2}, {58.0f, 64.0f, 139.0f, 154.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, Matmul2D2DMismatchedDimensions) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b = createTensor({2, 2}, {7.0f, 8.0f, 9.0f, 10.0f});
    ASSERT_THROW(matmul(a, b), std::invalid_argument);
}

// TEST 2D x 3D

TEST_F(TensorAdditionalOperationsTest, Matmul2D3D) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({2, 2, 2}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    Tensor result = matmul(a, b);

    Tensor expected = createTensor({2, 2, 2}, {31.0f, 34.0f, 71.0f, 78.0f, 43.0f, 46.0f, 99.0f, 106.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

// TEST 3D x 3D

TEST_F(TensorAdditionalOperationsTest, Matmul3D3D) {
    Tensor a = createTensor({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor b = createTensor({2, 2, 2}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    Tensor result = matmul(a, b);

    Tensor expected = createTensor({2, 2, 2}, {31.0f, 34.0f, 71.0f, 78.0f, 155.0f, 166.0f, 211.0f, 226.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, MatmulBatchedDataTest) {
    Tensor a = createTensor({4, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor b = createTensor({3, 2}, {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f});
    Tensor result = matmul(a, b);

    Tensor expected = createTensor({4, 2}, {94.0f, 100.0f, 229.0f, 244.0f, 364.0f, 388.0f, 499.0f, 532.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

// TEST BATCHED MATMUL

TEST_F(TensorAdditionalOperationsTest, BatchedMatmul) {
    // Case 1: (2, 3, 4) * (2, 4, 5)
    {
        Tensor a = createTensor({2, 3, 4}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        });
        Tensor b = createTensor({2, 4, 5}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
        });
        Tensor result = matmul(a, b);

        Tensor expected = createTensor({2, 3, 5}, {
            110, 120, 130, 140, 150, 246, 272, 298, 324, 350, 382,
            424, 466, 508, 550, 1678, 1736, 1794, 1852, 1910, 2134,
            2208, 2282, 2356, 2430, 2590, 2680, 2770, 2860, 2950
        });

        for (size_t i = 0; i < result.shape().size(); ++i) {
            EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
        }
    }

    // Case 2: (3, 2, 3) * (3, 3, 2)
    {
        Tensor a = createTensor({3, 2, 3}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18
        });
        Tensor b = createTensor({3, 3, 2}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
        });
        Tensor result = matmul(a, b);

        Tensor expected = createTensor({3, 2, 2}, {
            22, 28, 49, 64, 220, 244, 301, 334, 634, 676, 769, 820
        });

        for (size_t i = 0; i < result.shape().size(); ++i) {
            EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
        }
    }

    // Case 3: (2, 2, 2) * (2, 2)
    {
        Tensor a = createTensor({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
        Tensor b = createTensor({2, 2}, {9.0f, 10.0f, 11.0f, 12.0f});
        Tensor result = matmul(a, b);

        Tensor expected = createTensor({2, 2, 2}, {31.0f, 34.0f, 71.0f, 78.0f, 111.0f, 122.0f, 151.0f, 166.0f});

        for (size_t i = 0; i < result.shape().size(); ++i) {
            EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
        }
    }
}

/*

    TEST GENERATION FUNCTIONS

*/

TEST_F(TensorAdditionalOperationsTest, ZerosFunction) {
    Tensor result = zeros({6}, dtype::f32);

    Tensor expected = createTensor({6}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, ZerosFunction3D) {
    Tensor result = zeros({2, 2, 2}, dtype::f32);

    Tensor expected = createTensor({8}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, OnesFunction) {
    Tensor result = ones({2, 3}, dtype::f32);

    Tensor expected = createTensor({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

/*

    TEST SLICE FUNCTION

*/

TEST_F(TensorAdditionalOperationsTest, SliceFunction) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor result = a.slice({{0, 1}, {0, 1}});

    Tensor expected = createTensor({1, 1}, {1.0f});
    EXPECT_NEAR(result.at<float>(0), expected.at<float>(0), 1e-5);
}

TEST_F(TensorAdditionalOperationsTest, SliceFunctionOutOfBounds) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ASSERT_THROW(a.slice({{0, 3}, {0, 3}}), std::out_of_range);
}

TEST_F(TensorAdditionalOperationsTest, SliceFunctionInvalidRange) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ASSERT_THROW(a.slice({{0, 1}, {0, 3}}), std::out_of_range);
}

TEST_F(TensorAdditionalOperationsTest, ChangingSliceAffectsOriginalTensor) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor slice = a.slice({{0, 1}, {0, 1}});
    slice.set({0, 0}, 10.0f);

    Tensor expected = createTensor({2, 2}, {10.0f, 2.0f, 3.0f, 4.0f});
    for (size_t i = 0; i < a.shape().size(); ++i) {
        EXPECT_NEAR(a.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, MatrixMultiplicationBetweenTwoSlices) {
    Tensor a = createTensor({4, 4}, {1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f,
                                     9.0f, 10.0f, 11.0f, 12.0f,
                                     13.0f, 14.0f, 15.0f, 16.0f});

    Tensor b = createTensor({4, 4}, {16.0f, 15.0f, 14.0f, 13.0f,
                                     12.0f, 11.0f, 10.0f, 9.0f,
                                     8.0f, 7.0f, 6.0f, 5.0f,
                                     4.0f, 3.0f, 2.0f, 1.0f});

    // Slicing a 2x2 matrix from both tensors
    Tensor a_slice = a.slice({{2, 4}, {0, 2}});
    Tensor b_slice = b.slice({{2, 4}, {0, 2}});
    // Perform matrix multiplication between the slices
    Tensor result = matmul(a_slice, b_slice);
    
    // Expected result
    Tensor expected = createTensor({2, 2}, {112.0f, 93.0f,
                                            160.0f, 133.0f});
    
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, ElementWiseAdditionBetweenSlices) {
    Tensor a = createTensor({3, 3}, {1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f,
                                     7.0f, 8.0f, 9.0f});

    Tensor b = createTensor({3, 3}, {9.0f, 8.0f, 7.0f,
                                     6.0f, 5.0f, 4.0f,
                                     3.0f, 2.0f, 1.0f});

    // Slicing a 2x2 submatrix
    Tensor a_slice = a.slice({{1, 3}, {1, 3}});
    Tensor b_slice = b.slice({{1, 3}, {1, 3}});

    // Element-wise addition of slices
    Tensor result = a_slice + b_slice;

    // Expected result
    Tensor expected = createTensor({2, 2}, {10.0f, 10.0f,
                                            10.0f, 10.0f});

    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, NoMemoryOverlapBetweenSlices) {
    Tensor a = createTensor({4, 4}, {1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f,
                                     9.0f, 10.0f, 11.0f, 12.0f,
                                     13.0f, 14.0f, 15.0f, 16.0f});

    // Slicing different submatrices
    Tensor slice1 = a.slice({{0, 2}, {0, 2}});
    Tensor slice2 = a.slice({{2, 4}, {2, 4}});

    // Modifying the first slice
    slice1.set({0, 0}, 100.0f);

    // Check that modifying slice1 doesn't affect slice2
    Tensor expected_slice2 = createTensor({2, 2}, {11.0f, 12.0f,
                                                   15.0f, 16.0f});

    for (size_t i = 0; i < slice2.shape().size(); ++i) {
        EXPECT_NEAR(slice2.at<float>(i), expected_slice2.at<float>(i), 1e-5);
    }
}

/*

    TEST MAX FUNCTION

*/

TEST_F(TensorAdditionalOperationsTest, MaxFunctionNoAxis) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor result = max(a);

    EXPECT_EQ(result.shape(), Shape({1}));
    EXPECT_NEAR(result.at<float>(0), 6.0f, 1e-5);
}

TEST_F(TensorAdditionalOperationsTest, MaxFunctionAxis0) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor result = max(a, {0});

    Tensor expected = createTensor({3}, {4.0f, 5.0f, 6.0f});
    EXPECT_EQ(result.shape(), Shape({3}));
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, MaxFunctionAxis1) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor result = max(a, {1});

    Tensor expected = createTensor({2, 1}, {3.0f, 6.0f});
    EXPECT_EQ(result.shape(), Shape({2}));
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, MaxFunction3D) {
    Tensor a = createTensor({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor result = max(a, {1, 2});

    Tensor expected = createTensor({2}, {4.0f, 8.0f});
    EXPECT_EQ(result.shape(), Shape({2}));
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

/*

    TEST MIN FUNCTION

*/

TEST_F(TensorAdditionalOperationsTest, MinFunctionNoAxis) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor result = min(a);

    EXPECT_EQ(result.shape(), Shape({1}));
    EXPECT_NEAR(result.at<float>(0), 1.0f, 1e-5);
}

TEST_F(TensorAdditionalOperationsTest, MinFunctionAxis0) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor result = min(a, {0});

    Tensor expected = createTensor({3}, {1.0f, 2.0f, 3.0f});
    EXPECT_EQ(result.shape(), Shape({3}));
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, MinFunctionAxis1) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor result = min(a, {1});

    Tensor expected = createTensor({2, 1}, {1.0f, 4.0f});
    EXPECT_EQ(result.shape(), Shape({2}));
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

TEST_F(TensorAdditionalOperationsTest, MinFunction3D) {
    Tensor a = createTensor({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor result = min(a, {1, 2});

    Tensor expected = createTensor({2}, {1.0f, 5.0f});
    EXPECT_EQ(result.shape(), Shape({2}));
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected.at<float>(i), 1e-5);
    }
}

/*

    TEST MAX AND MIN TOGETHER

*/

TEST_F(TensorAdditionalOperationsTest, MaxMinFunctionComparison) {
    Tensor a = createTensor({2, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
    Tensor max_result = max(a);
    Tensor min_result = min(a);

    EXPECT_EQ(max_result.shape(), Shape({1}));
    EXPECT_EQ(min_result.shape(), Shape({1}));
    EXPECT_NEAR(max_result.at<float>(0), 4.0f, 1e-5);
    EXPECT_NEAR(min_result.at<float>(0), -1.0f, 1e-5);
}

TEST_F(TensorAdditionalOperationsTest, MaxMinFunctionWithKeepDims) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor max_result = max(a, {1}, true);
    Tensor min_result = min(a, {1}, true);

    EXPECT_EQ(max_result.shape(), Shape({2, 1}));
    EXPECT_EQ(min_result.shape(), Shape({2, 1}));
    
    Tensor max_expected = createTensor({2, 1}, {3.0f, 6.0f});
    Tensor min_expected = createTensor({2, 1}, {1.0f, 4.0f});
    
    for (size_t i = 0; i < max_result.shape().size(); ++i) {
        EXPECT_NEAR(max_result.at<float>(i), max_expected.at<float>(i), 1e-5);
        EXPECT_NEAR(min_result.at<float>(i), min_expected.at<float>(i), 1e-5);
    }
}

}  // namespace sdnn

#ifndef TEST_TENSOR_OPERATIONS_CPP
#define TEST_TENSOR_OPERATIONS_CPP

#include "../smart_dnn/TensorOperations.hpp"
#include "../utils/tensor_helpers.hpp"

/*

    TESTING ones FUNCTION

*/

TEST(TensorOperationsTest, OnesFunction_ValidShape) {
    Tensor a = TensorOperations::ones(2, 3);
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateTensorData(a, std::vector<float>(6, 1.0f));
}

TEST(TensorOperationsTest, OnesFunction_ZeroDimensions) {
    Tensor a = TensorOperations::ones(0, 0);
    ValidateTensorShape(a, 2, 0, {0, 0});
    ValidateTensorData(a, std::vector<float>(0));
}

/*

    TESTING identity FUNCTION

*/

TEST(TensorOperationsTest, IdentityFunction_ValidSize) {
    Tensor a = TensorOperations::identity(3);
    ValidateTensorShape(a, 2, 9, {3, 3});
    ValidateTensorData(a, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f});
}

TEST(TensorOperationsTest, IdentityFunction_SizeOne) {
    Tensor a = TensorOperations::identity(1);
    ValidateTensorShape(a, 2, 1, {1, 1});
    ValidateTensorData(a, {1.0f});
}

TEST(TensorOperationsTest, IdentityFunction_SizeZero) {
    Tensor a = TensorOperations::identity(0);
    ValidateTensorShape(a, 2, 0, {0, 0});
    ValidateTensorData(a, std::vector<float>(0));
}

/*

    TESTING randomn FUNCTION

*/

TEST(TensorOperationsTest, RandomnFunction_ValidShape) {
    Tensor a = TensorOperations::randomn({2, 3});
    ValidateTensorShape(a, 2, 6, {2, 3});

    for (auto value : a.getData()) {
        EXPECT_GE(value, -3.0f);
        EXPECT_LE(value, 3.0f);
    }
}

/*

    TESTING transpose FUNCTION

*/

TEST(TensorOperationsTest, TransposeFunction_ValidTranspose) {
    Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b = TensorOperations::transpose(a, 0, 1);

    ValidateTensorShape(b, 2, 6, {3, 2});
    ValidateTensorData(b, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});
}

TEST(TensorOperationsTest, TransposeFunction_InvalidTranspose) {
    Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    ASSERT_THROW(TensorOperations::transpose(a, 0, 2), std::out_of_range);
}

/*

    TESTING flattenIndex FUNCTION

*/

TEST(TensorOperationsTest, FlattenIndex_ValidInput) {
    std::vector<int> indices = {1, 2};
    Shape shape = {3, 4};

    int flatIndex = TensorOperations::flattenIndex(indices, shape);
    ASSERT_EQ(flatIndex, 6);
}

TEST(TensorOperationsTest, FlattenIndex_SingleElementShape) {
    std::vector<int> indices = {2};
    Shape shape = {4};

    int flatIndex = TensorOperations::flattenIndex(indices, shape);
    ASSERT_EQ(flatIndex, 2);
}

TEST(TensorOperationsTest, FlattenIndex_MismatchDimensions) {
    std::vector<int> indices = {1, 2, 3};
    Shape shape = {4, 5};

    ASSERT_THROW(TensorOperations::flattenIndex(indices, shape), std::invalid_argument);
}

/*

    TESTING getIndices FUNCTION

*/

TEST(TensorOperationsTest, GetIndices_ValidInput) {
    int flatIndex = 6;
    Shape shape = {3, 4};

    std::vector<int> indices = TensorOperations::getIndices(flatIndex, shape);
    ASSERT_EQ(indices.size(), 2);
    ASSERT_EQ(indices[0], 1);
    ASSERT_EQ(indices[1], 2);
}

TEST(TensorOperationsTest, GetIndices_SingleElementShape) {
    int flatIndex = 2;
    Shape shape = {4};

    std::vector<int> indices = TensorOperations::getIndices(flatIndex, shape);
    ASSERT_EQ(indices.size(), 1);
    ASSERT_EQ(indices[0], 2);
}

TEST(TensorOperationsTest, GetIndices_OutOfBoundIndex) {
    int flatIndex = 15;
    Shape shape = {3, 4};

    ASSERT_THROW(TensorOperations::getIndices(flatIndex, shape), std::out_of_range);
}

/*

    TESTING matmul FUNCTION

*/

TEST(TensorOperationsTest, Matmul_1D1DVectorDotProduct) {
    Tensor a({3}, {1.0f, 2.0f, 3.0f});
    Tensor b({3}, {4.0f, 5.0f, 6.0f});

    Tensor result = TensorOperations::matmul(a, b);
    ValidateTensorShape(result, 1, 1, {1});
    ValidateTensorData(result, {32.0f});
}

TEST(TensorOperationsTest, Matmul_2D1DMatrixVectorProduct) {
    Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b({3}, {7.0f, 8.0f, 9.0f});

    Tensor result = TensorOperations::matmul(a, b);
    ValidateTensorShape(result, 1, 2, {2});
    ValidateTensorData(result, {50.0f, 122.0f});
}

TEST(TensorOperationsTest, Matmul_2D2DMatrixProduct) {
    Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    Tensor result = TensorOperations::matmul(a, b);
    ValidateTensorShape(result, 2, 4, {2, 2});
    ValidateTensorData(result, {58.0f, 64.0f, 139.0f, 154.0f});
}

TEST(TensorOperationsTest, Matmul_3DMatrixMultiplication) {
    Tensor a({2, 3, 4}, TensorOperations::ones(2, 3, 4).getData());
    Tensor b({2, 4, 5}, TensorOperations::ones(2, 4, 5).getData());

    Tensor result = TensorOperations::matmul(a, b);
    ValidateTensorShape(result, 3, 30, {2, 3, 5});
    ValidateTensorData(result, std::vector<float>(30, 4.0f));
}

TEST(TensorOperationsTest, Matmul_InvalidShapeFor1D1D) {
    Tensor a({3}, {1.0f, 2.0f, 3.0f});
    Tensor b({4}, {4.0f, 5.0f, 6.0f, 7.0f});

    ASSERT_THROW(TensorOperations::matmul(a, b), std::invalid_argument);
}

TEST(TensorOperationsTest, Matmul_InvalidShapeFor2D2D) {
    Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b({2, 2}, {7.0f, 8.0f, 9.0f, 10.0f});

    ASSERT_THROW(TensorOperations::matmul(a, b), std::invalid_argument);
}


#endif // TEST_TENSOR_OPERATIONS_CPP
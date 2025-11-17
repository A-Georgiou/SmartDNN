#include <gtest/gtest.h>
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"

namespace sdnn {

class ArrayFireBackendTest : public ::testing::Test {
protected:
    Tensor createTensor(const std::vector<int>& shape_, const std::vector<float>& values) {
        return Tensor(Shape(shape_), values);
    }
};

TEST_F(ArrayFireBackendTest, BasicTensorCreation) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    EXPECT_EQ(a.shape().rank(), 2);
    EXPECT_EQ(a.shape()[0], 2);
    EXPECT_EQ(a.shape()[1], 2);
    EXPECT_EQ(a.shape().size(), 4);
}

TEST_F(ArrayFireBackendTest, TensorAddition) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    
    Tensor result = a + b;
    
    EXPECT_EQ(result.shape().rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_F(ArrayFireBackendTest, TensorMultiplication) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
    
    Tensor result = a * b;
    
    EXPECT_EQ(result.shape().rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_F(ArrayFireBackendTest, ScalarAddition) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    Tensor result = a + 10.0f;
    
    EXPECT_EQ(result.shape().rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_F(ArrayFireBackendTest, MatrixMultiplication) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b = createTensor({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    
    Tensor result = matmul(a, b);
    
    EXPECT_EQ(result.shape().rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_F(ArrayFireBackendTest, TensorTranspose) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    Tensor result = transpose(a, {1, 0});
    
    EXPECT_EQ(result.shape().rank(), 2);
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_F(ArrayFireBackendTest, SumReduction) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    Tensor result = sum(a, {0}, false);
    
    EXPECT_EQ(result.shape().rank(), 1);
    EXPECT_EQ(result.shape()[0], 3);
}

TEST_F(ArrayFireBackendTest, MeanReduction) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    Tensor result = mean(a, {0}, false);
    
    EXPECT_EQ(result.shape().rank(), 1);
    EXPECT_EQ(result.shape()[0], 3);
}

TEST_F(ArrayFireBackendTest, ExpFunction) {
    Tensor a = createTensor({2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
    
    Tensor result = exp(a);
    
    EXPECT_EQ(result.shape().rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_F(ArrayFireBackendTest, LogFunction) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    Tensor result = log(a);
    
    EXPECT_EQ(result.shape().rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_F(ArrayFireBackendTest, SqrtFunction) {
    Tensor a = createTensor({2, 2}, {1.0f, 4.0f, 9.0f, 16.0f});
    
    Tensor result = sqrt(a);
    
    EXPECT_EQ(result.shape().rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_F(ArrayFireBackendTest, ReciprocalFunction) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 4.0f, 5.0f});
    
    Tensor result = reciprocal(a);
    
    EXPECT_EQ(result.shape().rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
}

TEST_F(ArrayFireBackendTest, BackendName) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    std::string backendName = a.backend().backendName();
    
    EXPECT_TRUE(backendName.find("GPU") != std::string::npos || 
                backendName.find("ArrayFire") != std::string::npos);
}

} // namespace sdnn

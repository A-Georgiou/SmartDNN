#include <gtest/gtest.h>
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"

namespace sdnn {

class TensorOperationsTest : public ::testing::Test {
protected:
    Tensor createTensor(const std::vector<int>& shape_, const std::vector<float>& values) {
        return Tensor(Shape(shape_), values);
    }
};

TEST_F(TensorOperationsTest, AddTwoTensors) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

    Tensor result = a + b;

    std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, SubtractTwoTensors) {
    Tensor a = createTensor({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor b = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor result = a - b;

    std::vector<float> expected = {4.0f, 4.0f, 4.0f, 4.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, MultiplyTwoTensors) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

    Tensor result = a * b;

    std::vector<float> expected = {5.0f, 12.0f, 21.0f, 32.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, DivideTwoTensors) {
    Tensor a = createTensor({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    Tensor b = createTensor({2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});

    Tensor result = a / b;

    std::vector<float> expected = {5.0f, 5.0f, 5.0f, 5.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, AddScalarToTensor) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    double scalar = 1.5;

    Tensor result = a + scalar;

    std::vector<float> expected = {2.5f, 3.5f, 4.5f, 5.5f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, SumOverAxes) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor result = sum(a, {1}, false);

    std::vector<float> expected = {3.0f, 7.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, MeanOverAxes) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor result = mean(a, {1}, false);

    std::vector<float> expected = {1.5f, 3.5f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, ReshapeTensor) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor result = reshape(a, {4});

    EXPECT_EQ(result.shape().rank(), 1);
    EXPECT_EQ(result.shape()[0], 4);
}

TEST_F(TensorOperationsTest, TransposeTensor) {
    Tensor a = createTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Tensor result = transpose(a, {1, 0});

    std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, ElementWiseExp) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor result = exp(a);

    std::vector<float> expected = {std::exp(1.0f), std::exp(2.0f), std::exp(3.0f), std::exp(4.0f)};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, ElementWiseSqrt) {
    Tensor a = createTensor({2, 2}, {1.0f, 4.0f, 9.0f, 16.0f});

    Tensor result = sqrt(a);

    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, ElementWiseLog) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.71828f, 7.38906f, 20.0855f});

    Tensor result = log(a);

    std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TensorOperationsTest, ScalarOperations) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor result = a + 1.0;
    std::vector<float> expectedAdd = {2.0f, 3.0f, 4.0f, 5.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expectedAdd[i], 1e-5);
    }

    result = a - 1.0;
    std::vector<float> expectedSub = {0.0f, 1.0f, 2.0f, 3.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expectedSub[i], 1e-5);
    }

    result = a * 2.0;
    std::vector<float> expectedMul = {2.0f, 4.0f, 6.0f, 8.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expectedMul[i], 1e-5);
    }

    result = a / 2.0;
    std::vector<float> expectedDiv = {0.5f, 1.0f, 1.5f, 2.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expectedDiv[i], 1e-5);
    }
}



} // namespace sdnn

#include <gtest/gtest.h>
#include "../smart_dnn/Tensor.hpp"
#include "../smart_dnn/TensorOperations.hpp"

/*

    HELPER FUNCTIONS

*/

void ValidateTensorShape(const Tensor& tensor, int rank, int size, const std::vector<int>& dimensions) {
    ASSERT_EQ(tensor.shape().rank(), rank);
    ASSERT_EQ(tensor.shape().size(), size);
    for (int i = 0; i < rank; ++i) {
        ASSERT_EQ(tensor.shape()[i], dimensions[i]);
    }
}

void ValidateTensorData(const Tensor& tensor, const std::vector<float>& expectedData) {
    ASSERT_EQ(tensor.getData().size(), expectedData.size());
    for (size_t i = 0; i < expectedData.size(); ++i) {
        ASSERT_FLOAT_EQ(tensor.getData()[i], expectedData[i]);
    }
}


/*

    VALID INITIALISATION TESTS

*/

TEST(TensorTest, ExpectDefaultShapeInitialisation) {
    Tensor a({1, 2, 3});

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 0.0f));
}

TEST(TensorTest, ExpectValueFillInitialisation) {
    Tensor a({1, 2, 3}, 5.0f);

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 5.0f));
}

TEST(TensorTest, ExpectValidDataInitialisation) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor a({1, 2, 3}, data);
    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, data);
}

TEST(TensorTest, ExpectValidTensorCopy) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor a({1, 2, 3}, data);
    Tensor b(a);

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, data);
}

TEST(TensorTest, ExpectValidTensorMove) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor a({1, 2, 3}, data);
    Tensor b(std::move(a));

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, data);
}

/*

    INVALID INITIALISATION TESTS

*/

TEST(TensorTest, ExpectInvalidShapeInitialisation) {
    ASSERT_THROW(Tensor a({-1, -1, -1}), std::invalid_argument);
    ASSERT_THROW(Tensor b({1,1,1}, {1.0f, 1.0f}), std::invalid_argument);
}

/*

    VALID OPERATOR TESTS

*/

TEST(TensorTest, ExpectValidElementAccess) {
    Tensor a({1, 2, 3}, 5.0f);

    ASSERT_FLOAT_EQ(a({0, 0, 0}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 0, 1}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 0, 2}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 1, 0}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 1, 1}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 1, 2}), 5.0f);
}

TEST(TensorTest, ExpectValidElementAssignment) {
    Tensor a({1, 2, 3}, 5.0f);

    a({0, 0, 0}) = 1.0f;
    a({0, 0, 1}) = 2.0f;
    a({0, 0, 2}) = 3.0f;
    a({0, 1, 0}) = 4.0f;
    a({0, 1, 1}) = 5.0f;
    a({0, 1, 2}) = 6.0f;

    ASSERT_FLOAT_EQ(a({0, 0, 0}), 1.0f);
    ASSERT_FLOAT_EQ(a({0, 0, 1}), 2.0f);
    ASSERT_FLOAT_EQ(a({0, 0, 2}), 3.0f);
    ASSERT_FLOAT_EQ(a({0, 1, 0}), 4.0f);
    ASSERT_FLOAT_EQ(a({0, 1, 1}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 1, 2}), 6.0f);
}

TEST(TensorTest, ExpectValidElementWiseAddition) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);
    Tensor c = a + b;

    ValidateTensorShape(c, 3, 6, {1, 2, 3});
    ValidateTensorData(c, std::vector<float>(6, 10.0f));
}

TEST(TensorTest, ExpectValidElementWiseSubtraction) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({2, 3}, 5.0f);

    Tensor c = a - b;

    ValidateTensorShape(c, 3, 6, {1, 2, 3});
    ValidateTensorData(c, std::vector<float>(6, 0.0f));
}

TEST(TensorTest, ExpectValidElementWiseMultiplication) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    Tensor c = a * b;

    ValidateTensorShape(c, 3, 6, {1, 2, 3});
    ValidateTensorData(c, std::vector<float>(6, 25.0f));
}

TEST(TensorTest, ExpectValidElementWiseDivision) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    Tensor c = a / b;

    ValidateTensorShape(c, 3, 6, {1, 2, 3});
    ValidateTensorData(c, std::vector<float>(6, 1.0f));
}

TEST(TensorTest, ExpectValidElementWiseAdditionAssignment) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    a += b;

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 10.0f));
}

TEST(TensorTest, ExpectValidElementWiseSubtractionAssignment) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    a -= b;

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 0.0f));
}

TEST(TensorTest, ExpectValidElementWiseMultiplicationAssignment) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    a *= b;

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 25.0f));
}

TEST(TensorTest, ExpectValidElementWiseDivisionAssignment) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    a /= b;

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 1.0f));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#include <gtest/gtest.h>
#include "../smart_dnn/Tensor.hpp"

/*

    VALID INITIALISATION TESTS

*/

TEST(TensorTest, ExpectDefaultShapeInitialisation) {
    Tensor a({1, 2, 3});

    ASSERT_EQ(a.shape().rank(), 3);
    ASSERT_EQ(a.shape().size(), 6);
    ASSERT_EQ(a.shape()[0], 1);
    ASSERT_EQ(a.shape()[1], 2);
    ASSERT_EQ(a.shape()[2], 3);
    ASSERT_EQ(a.getData().size(), 6);
}

TEST(TensorTest, ExpectValueFillInitialisation) {
    Tensor a({1, 2, 3}, 5.0f);

    ASSERT_EQ(a.shape().rank(), 3);
    ASSERT_EQ(a.shape().size(), 6);
    ASSERT_EQ(a.shape()[0], 1);
    ASSERT_EQ(a.shape()[1], 2);
    ASSERT_EQ(a.shape()[2], 3);
    ASSERT_EQ(a.getData().size(), 6);
    for (int i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(a.getData()[i], 5.0f);
    }
}

TEST(TensorTest, ExpectValidDataInitialisation) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor a({1, 2, 3}, data);

    ASSERT_EQ(a.shape().rank(), 3);
    ASSERT_EQ(a.shape().size(), 6);
    ASSERT_EQ(a.shape()[0], 1);
    ASSERT_EQ(a.shape()[1], 2);
    ASSERT_EQ(a.shape()[2], 3);
    ASSERT_EQ(a.getData().size(), 6);
    for (int i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(a.getData()[i], data[i]);
    }
}

TEST(TensorTest, ExpectValidTensorCopy) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor a({1, 2, 3}, data);
    Tensor b(a);

    ASSERT_EQ(b.shape().rank(), 3);
    ASSERT_EQ(b.shape().size(), 6);
    ASSERT_EQ(b.shape()[0], 1);
    ASSERT_EQ(b.shape()[1], 2);
    ASSERT_EQ(b.shape()[2], 3);
    ASSERT_EQ(b.getData().size(), 6);
    for (int i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(b.getData()[i], data[i]);
    }
}

TEST(TensorTest, ExpectValidTensorMove) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor a({1, 2, 3}, data);
    Tensor b(std::move(a));

    ASSERT_EQ(b.shape().rank(), 3);
    ASSERT_EQ(b.shape().size(), 6);
    ASSERT_EQ(b.shape()[0], 1);
    ASSERT_EQ(b.shape()[1], 2);
    ASSERT_EQ(b.shape()[2], 3);
    ASSERT_EQ(b.getData().size(), 6);
    for (int i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(b.getData()[i], data[i]);
    }
}

/*

    INVALID INITIALISATION TESTS

*/

TEST(TensorTest, ExpectInvalidShapeInitialisation) {
    ASSERT_THROW(Tensor a({-1, -1, -1}), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#ifndef TEST_TENSOR_CPP
#define TEST_TENSOR_CPP

#include "../utils/tensor_helpers.hpp"

/*

    VALID INITIALISATION TESTS

*/

namespace smart_dnn {

TEST(TensorInitialisationTest, ExpectDefaultShapeInitialisation) {
    Tensor a({1, 2, 3});

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 0.0f));
}

TEST(TensorInitialisationTest, ExpectValueFillInitialisation) {
    Tensor a({1, 2, 3}, 5.0f);

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 5.0f));
}

TEST(TensorInitialisationTest, ExpectVectorInitialisation) {
    const std::vector<float> values = {1, 2, 3, 4, 5, 6};
    Tensor<float> copied(Shape({2, 3}), values);
    Tensor<float> moved(Shape({2, 3}), std::vector<float>(values));
    Tensor<float> shorter(Shape({2, 4}), values);

    ValidateTensorShape(copied, 2, 6, {2, 3});
    ValidateTensorShape(moved, 2, 6, {2, 3});
    ValidateTensorData(copied, values);
    ValidateTensorData(moved, values);
    ValidateTensorData(shorter, {1, 2, 3, 4, 5, 6, 0, 0});
}

TEST(TensorInitialisationTest, ExpectOversizedValuesRejected) {
    const std::vector<float> values = {1, 2, 3};
    EXPECT_THROW(Tensor<float>(Shape({2}), values), std::invalid_argument);
    EXPECT_THROW(Tensor<float>(Shape({2}), std::vector<float>(values)), std::invalid_argument);
    EXPECT_THROW(Tensor<float>(Shape({2}), {1.0f, 2.0f, 3.0f}), std::invalid_argument);
}

/*

    INVALID INITIALISATION TESTS

*/

TEST(TensorInitialisationTest, ExpectInvalidShapeInitialisation) {
    ASSERT_THROW(Tensor a({-1, -1, -1}), std::invalid_argument);
}

/*

    VALID OPERATOR TESTS

*/

TEST(TensorOperatorTest, ExpectValidElementWiseAddition) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);
    Tensor c = a + b;

    ValidateTensorShape(c, 3, 6, {1, 2, 3});
    ValidateTensorData(c, std::vector<float>(6, 10.0f));
}

TEST(TensorOperatorTest, EqualShapeAdditionPreservesValuesAndIndependentStorage) {
    Tensor<float> a({2, 2}, 0.0f);
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    a[3] = 4.0f;
    Tensor<float> sum = a + a;

    ValidateTensorShape(sum, 2, 4, {2, 2});
    ValidateTensorData(sum, std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f});
    EXPECT_NE(sum.getData().data(), a.getData().data());

    sum[0] = 100.0f;
    EXPECT_EQ(a[0], 1.0f);
}

TEST(TensorOperatorTest, EqualShapeAdditionSupportsDoubleAndIntegerTensors) {
    Tensor<double> doubles({2}, 0.0);
    doubles[0] = 1.5;
    doubles[1] = 2.5;
    Tensor<double> doubleSum = doubles + doubles;
    ValidateTensorShape(doubleSum, 1, 2, {2});
    ValidateTensorData(doubleSum, std::vector<double>{3.0, 5.0});

    Tensor<int> integers({1, 2, 2}, 0);
    integers[0] = 1;
    integers[1] = 2;
    integers[2] = 3;
    integers[3] = 4;
    Tensor<int> integerSum = integers + integers;
    ValidateTensorShape(integerSum, 3, 4, {1, 2, 2});
    ValidateTensorData(integerSum, std::vector<int>{2, 4, 6, 8});
}

TEST(TensorOperatorTest, EqualElementCountsWithDifferentShapesStillBroadcast) {
    Tensor<float> row({1, 2}, 0.0f);
    row[0] = 1.0f;
    row[1] = 2.0f;
    Tensor<float> column({2, 1}, 0.0f);
    column[0] = 10.0f;
    column[1] = 20.0f;
    Tensor<float> sum = row + column;

    ValidateTensorShape(sum, 2, 4, {2, 2});
    ValidateTensorData(sum, std::vector<float>{11.0f, 12.0f, 21.0f, 22.0f});
}

TEST(TensorOperatorTest, ExpectValidElementWiseSubtraction) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({2, 3}, 5.0f);

    Tensor c = a - b;

    ValidateTensorShape(c, 3, 6, {1, 2, 3});
    ValidateTensorData(c, std::vector<float>(6, 0.0f));
}

TEST(TensorOperatorTest, ExpectValidElementWiseMultiplication) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    Tensor c = a * b;

    ValidateTensorShape(c, 3, 6, {1, 2, 3});
    ValidateTensorData(c, std::vector<float>(6, 25.0f));
}

TEST(TensorOperatorTest, ExpectValidElementWiseDivision) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    Tensor c = a / b;

    ValidateTensorShape(c, 3, 6, {1, 2, 3});
    ValidateTensorData(c, std::vector<float>(6, 1.0f));
}

TEST(TensorOperatorTest, ExpectValidElementWiseAdditionAssignment) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    a += b;

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 10.0f));
}

TEST(TensorOperatorTest, ExpectValidElementWiseSubtractionAssignment) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    a -= b;

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 0.0f));
}

TEST(TensorOperatorTest, ExpectValidElementWiseMultiplicationAssignment) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    a *= b;

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 25.0f));
}

TEST(TensorOperatorTest, ExpectValidElementWiseDivisionAssignment) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);

    a /= b;

    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, std::vector<float>(6, 1.0f));
}

/*

    COPY AND MOVE TESTS

*/

TEST(TensorCopyMoveTest, SelfAssignment) {
    Tensor a({2, 3}, 5.0f);
    a = a; // Self-assignment

    // Ensure tensor remains unchanged
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateTensorData(a, std::vector<float>(6, 5.0f));
}

TEST(TensorCopyMoveTest, DoubleMove) {
    Tensor a({2, 3}, 5.0f);
    Tensor b(std::move(a)); // First move

    // Ensure b has the data
    ValidateTensorShape(b, 2, 6, {2, 3});
    ValidateTensorData(b, std::vector<float>(6, 5.0f));

    Tensor c(std::move(b)); // Second move
    ValidateTensorShape(c, 2, 6, {2, 3});
    ValidateTensorData(c, std::vector<float>(6, 5.0f));

    // b should be in a valid, but empty state.
    // We cannot have empty tensors so it is set to 1
    ASSERT_EQ(b.getShape().size(), 1);
}

TEST(TensorCopyMoveTest, MoveAssignmentAfterCopy) {
    Tensor a({2, 3}, 5.0f);
    Tensor b(a);           // Copy constructor
    Tensor c = std::move(a); // Move assignment

    // Ensure c has the data
    ValidateTensorShape(c, 2, 6, {2, 3});
    ValidateTensorData(c, std::vector<float>(6, 5.0f));

    // Assign b to a (which was moved from)
    a = b;

    // Ensure a now has b's data
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateTensorData(a, std::vector<float>(6, 5.0f));
}


/*

    VALID SCALAR OPERATOR TESTS

*/

TEST(TensorScalarOperatorTest, ExpectValidScalarAddition) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b = a + 5.0f;

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, std::vector<float>(6, 10.0f));
}

TEST(TensorScalarOperatorTest, ExpectValidScalarSubtraction) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b = a - 5.0f;

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, std::vector<float>(6, 0.0f));
}

TEST(TensorScalarOperatorTest, ExpectValidScalarMultiplication) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b = a * 5.0f;

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, std::vector<float>(6, 25.0f));
}

TEST(TensorScalarOperatorTest, ExpectValidScalarDivision) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b = a / 5.0f;

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, std::vector<float>(6, 1.0f));
}


/*

    VALID INVERSE SCALAR OPERATOR TESTS

*/

TEST(TensorScalarOperatorTest, ExpectValidScalarAdditionInverse) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b = 5.0f + a;

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, std::vector<float>(6, 10.0f));
}

TEST(TensorScalarOperatorTest, ExpectValidScalarSubtractionInverse) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b = 6.0f - a;

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, std::vector<float>(6, 1.0f));
}

TEST(TensorScalarOperatorTest, ExpectValidScalarMultiplicationInverse) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b = 5.0f * a;

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, std::vector<float>(6, 25.0f));
}

TEST(TensorScalarOperatorTest, ExpectValidScalarDivisionInverse) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b = 10.0f / a;

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, std::vector<float>(6, 2.0f));
}

// ELEMENT ACCESS TESTS

TEST(TensorAccessTest, MultiIndexAccessMatchesFlatLayout) {
    Tensor<float> t(Shape({2, 3, 4}));
    for (size_t i = 0; i < t.getShape().size(); ++i) t[i] = static_cast<float>(i);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k) {
                const float expected = static_cast<float>((i * 3 + j) * 4 + k);
                EXPECT_EQ(t(i, j, k), expected);
                EXPECT_EQ(t.at({i, j, k}), expected);
                EXPECT_EQ(t.at(std::vector<int>{i, j, k}), expected);
            }

    t(1, 2, 3) = -1.0f;
    EXPECT_EQ(t.at({1, 2, 3}), -1.0f);
    EXPECT_EQ(computeFlatIndex(t.getShape(), {1, 2, 3}), 23u);
}

TEST(TensorAccessTest, AtChecksRankAndBounds) {
    Tensor<float> t(Shape({2, 3}));
    EXPECT_THROW(t.at({1, 3}), std::out_of_range);
    EXPECT_THROW(t.at({-1, 0}), std::out_of_range);
    EXPECT_THROW(t.at({1}), std::invalid_argument);
    EXPECT_THROW(t.at(std::vector<int>{0, 0, 0}), std::invalid_argument);
}

} // namespace smart_dnn

#endif // TEST_TENSOR_CPP
#ifndef TEST_TENSOR_CPP
#define TEST_TENSOR_CPP

#include "tensor_helpers.hpp"

/*

    VALID INITIALISATION TESTS

*/

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

TEST(TensorInitialisationTest, ExpectValidDataInitialisation) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor a({1, 2, 3}, data);
    ValidateTensorShape(a, 3, 6, {1, 2, 3});
    ValidateTensorData(a, data);
}

TEST(TensorInitialisationTest, ExpectValidTensorCopy) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor a({1, 2, 3}, data);
    Tensor b(a);

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, data);
}

TEST(TensorInitialisationTest, ExpectValidTensorMove) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor a({1, 2, 3}, data);
    Tensor b(std::move(a));

    ValidateTensorShape(b, 3, 6, {1, 2, 3});
    ValidateTensorData(b, data);
}

/*

    INVALID INITIALISATION TESTS

*/

TEST(TensorInitialisationTest, ExpectInvalidShapeInitialisation) {
    ASSERT_THROW(Tensor a({-1, -1, -1}), std::invalid_argument);
    ASSERT_THROW(Tensor b({1,1,1}, {1.0f, 1.0f}), std::invalid_argument);
}

/*

    VALID OPERATOR TESTS

*/

TEST(TensorOperatorTest, ExpectValidElementAccess) {
    Tensor a({1, 2, 3}, 5.0f);

    ASSERT_FLOAT_EQ(a({0, 0, 0}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 0, 1}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 0, 2}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 1, 0}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 1, 1}), 5.0f);
    ASSERT_FLOAT_EQ(a({0, 1, 2}), 5.0f);
}

TEST(TensorOperatorTest, ExpectValidElementAssignment) {
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

TEST(TensorOperatorTest, ExpectValidElementWiseAddition) {
    Tensor a({1, 2, 3}, 5.0f);
    Tensor b({1, 2, 3}, 5.0f);
    Tensor c = a + b;

    ValidateTensorShape(c, 3, 6, {1, 2, 3});
    ValidateTensorData(c, std::vector<float>(6, 10.0f));
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

TEST(TensorCopyMoveTest, CopyConstructorCreatesDeepCopy) {
    Tensor a({2, 3}, 5.0f);
    Tensor b(a); // Copy constructor

    // Modify b and check that a remains unchanged
    b({0, 0}) = 10.0f;
    ASSERT_FLOAT_EQ(b({0, 0}), 10.0f);
    ASSERT_FLOAT_EQ(a({0, 0}), 5.0f);
}

TEST(TensorCopyMoveTest, CopyAssignmentCreatesDeepCopy) {
    Tensor a({2, 3}, 5.0f);
    Tensor b = a; // Copy assignment

    // Modify b and check that a remains unchanged
    b({0, 0}) = 10.0f;
    ASSERT_FLOAT_EQ(b({0, 0}), 10.0f);
    ASSERT_FLOAT_EQ(a({0, 0}), 5.0f);
}

TEST(TensorCopyMoveTest, MoveConstructorTransfersOwnership) {
    Tensor a({2, 3}, 5.0f);
    Tensor b(std::move(a)); // Move constructor

    // Ensure b has the data
    ValidateTensorShape(b, 2, 6, {2, 3});
    ValidateTensorData(b, std::vector<float>(6, 5.0f));

    // a should be in a valid, but empty state
    ASSERT_EQ(a.size().size(), 0);
}

TEST(TensorCopyMoveTest, MoveAssignmentTransfersOwnership) {
    Tensor a({2, 3}, 5.0f);
    Tensor b = std::move(a); // Move assignment

    // Ensure b has the data
    ValidateTensorShape(b, 2, 6, {2, 3});
    ValidateTensorData(b, std::vector<float>(6, 5.0f));

    // a should be in a valid, but empty state
    ASSERT_EQ(a.size().size(), 0);
}

TEST(TensorCopyMoveTest, CopyAndMoveLargeTensor) {
    std::vector<float> largeData(1000000, 1.0f);
    Tensor a({1000, 1000}, largeData);

    Tensor b(a); // Copy constructor
    ValidateTensorData(b, largeData);

    Tensor c(std::move(a)); // Move constructor
    ValidateTensorShape(c, 2, 1000000, {1000, 1000});
    ValidateTensorData(c, largeData);

    // a should be in a valid, but empty state
    ASSERT_EQ(a.size().size(), 0);
}

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

    // b should be in a valid, but empty state
    ASSERT_EQ(b.size().size(), 0);
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

#endif // TEST_TENSOR_CPP
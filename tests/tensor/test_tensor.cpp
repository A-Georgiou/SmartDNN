#ifndef TEST_TENSOR_CPP
#define TEST_TENSOR_CPP

#include "tests/utils/tensor_helpers.hpp"

/*

    VALID INITIALISATION TESTS

*/

namespace sdnn {

TEST(TensorInitialisationTest, ExpectInvalidShapeZeroElement) {
    // Expecting an exception when one dimension is zero, but others are not
    ASSERT_THROW(Tensor a({2, 0, 3}), std::invalid_argument);
}

TEST(TensorInitialisationTest, ExpectInvalidShapeNegativeElement) {
    // Expecting an exception for negative dimensions
    ASSERT_THROW(Tensor a({2, -1, 3}), std::invalid_argument);
}

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

/* INVALID OPERATOR TESTS */

TEST(TensorOperatorTest, ExpectIncompatibleShapeAddition) {
    Tensor a({2, 3}, 1.0f);
    Tensor b({2, 2}, 1.0f);

    ASSERT_THROW(Tensor c = a + b, std::invalid_argument);
}

TEST(TensorOperatorTest, ExpectIncompatibleShapeSubtraction) {
    Tensor a({2, 3}, 1.0f);
    Tensor b({2, 2}, 1.0f);

    ASSERT_THROW(Tensor c = a - b, std::invalid_argument);
}

TEST(TensorOperatorTest, ExpectIncompatibleShapeMultiplication) {
    Tensor a({2, 3}, 1.0f);
    Tensor b({2, 2}, 1.0f);

    ASSERT_THROW(Tensor c = a * b, std::invalid_argument);
}

TEST(TensorOperatorTest, ExpectIncompatibleShapeDivision) {
    Tensor a({2, 3}, 1.0f);
    Tensor b({2, 2}, 1.0f);

    ASSERT_THROW(Tensor c = a / b, std::invalid_argument);
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

TEST(TensorOperatorTest, ExpectTensorAdditionWithBroadcasting) {
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b({2}, {5.0f, 6.0f});

    Tensor c = a + b;

    ValidateTensorShape(c, 2, 4, {2, 2});
    ValidateTensorData(c, std::vector<float>{6.0f, 8.0f, 8.0f, 10.0f});
}

TEST(TensorOperatorTest, ExpectTensorSubtractionWithBroadcasting) {
    Tensor a({2, 2}, {5.0f, 7.0f, 9.0f, 11.0f});
    Tensor b({2}, {3.0f, 4.0f});

    Tensor c = a - b;

    ValidateTensorShape(c, 2, 4, {2, 2});
    ValidateTensorData(c, std::vector<float>{2.0f, 3.0f, 6.0f, 7.0f});
}

TEST(TensorOperatorTest, ExpectTensorMultiplicationWithBroadcasting) {
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b({2}, {10.0f, 20.0f});

    Tensor c = a * b;

    ValidateTensorShape(c, 2, 4, {2, 2});
    ValidateTensorData(c, std::vector<float>{10.0f, 40.0f, 30.0f, 80.0f});
}

TEST(TensorOperatorTest, ExpectTensorDivisionWithBroadcasting) {
    Tensor a({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    Tensor b({2}, {2.0f, 5.0f});

    Tensor c = a / b;

    ValidateTensorShape(c, 2, 4, {2, 2});
    ValidateTensorData(c, std::vector<float>{5.0f, 4.0f, 15.0f, 8.0f});
}

/*

    COPY AND MOVE TESTS

*/

TEST(TensorCopyMoveTest, ValidMoveAssignment) {
    Tensor a({2, 3}, 5.0f);
    Tensor b = std::move(a);  // Move assignment

    // Ensure b has the data
    ValidateTensorShape(b, 2, 6, {2, 3});
    ValidateTensorData(b, std::vector<float>(6, 5.0f));

    // a should now be empty or invalid (depending on your Tensor move semantics)
    // No need to check a, just b.
}

TEST(TensorCopyMoveTest, MoveConstructorAndAssignment) {
    Tensor a({2, 3}, 5.0f);

    Tensor b(std::move(a));  // Move constructor

    ValidateTensorShape(b, 2, 6, {2, 3});
    ValidateTensorData(b, std::vector<float>(6, 5.0f));

    Tensor c = std::move(b);  // Move assignment

    ValidateTensorShape(c, 2, 6, {2, 3});
    ValidateTensorData(c, std::vector<float>(6, 5.0f));
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


/*

VALID OPERATOR TESTS

*/

TEST(TensorOperatorTest, ExpectValidAccessOperator) {
    Tensor a({1, 2, 3}, 1.0f);
    
    for (size_t i = 0; i < 6; ++i) {
        Tensor b = a[i];
        b *= i;
    }

    std::vector<float> expectedData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ValidateTensorData(a, expectedData);
}

TEST(TensorOperatorTest, ExpectValidAccessOperatorMathWithTensor) {
    Tensor a({1, 2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor b({1}, 2.0f);
    
    for (size_t i = 0; i < 6; ++i) {
        Tensor c = a[i];
        c += b;
    }

    std::vector<float> expectedData = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    ValidateTensorData(a, expectedData);
}

/*

    ACCESS AND INDEXING TESTS

*/

TEST(TensorAccessTest, AccessSingleElement) {
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    float val = a.at<float>(1);

    EXPECT_NEAR(val, 2.0f, 1e-5);
}

TEST(TensorAccessTest, AccessMultipleElements) {
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    std::vector<size_t> indices = {1, 1};
    float val = a.at<float>(indices);

    EXPECT_NEAR(val, 4.0f, 1e-5);
}

TEST(TensorAccessTest, ModifySingleElement) {
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    a.set(1, 10.0f);

    EXPECT_NEAR(a.at<float>(1), 10.0f, 1e-5);
}

TEST(TensorAccessTest, ModifyMultipleElements) {
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    a.set({1, 1}, 10.0f);

    EXPECT_NEAR(a.at<float>({1, 1}), 10.0f, 1e-5);
}

TEST(TensorAccessTest, AccessOutOfBounds) {
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    ASSERT_THROW(a.at<float>({2, 2}), std::out_of_range);
}

TEST(TensorAccessTest, ModifyOutOfBounds) {
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    ASSERT_THROW(a.set({2, 2}, 10.0f), std::out_of_range);
}

/*

    TEST SUB-VIEW OPERATORS

*/

TEST(TensorSubViewTest, ExpectValidSingleElementAccess) {
    // 2x2x2 tensor
    Tensor a({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    // Accessing a single element using an initializer list
    Tensor view = a[2];  // This should return a 1-element sub-view

    ValidateTensorShape(view, 1, 1, {1});  // Scalar tensor view
    EXPECT_NEAR(view.at<float>(0), 3.0f, 1e-5);
}

TEST(TensorSubViewTest, ExpectValidSingleElementAccessUsingInitializerList) {
    // 2x2x2 tensor
    Tensor a({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    // Accessing a single element using an initializer list
    Tensor view = a[{1, 1, 0}];  // This should return a 1-element sub-view

    ValidateTensorShape(view, 1, 1, {1});  // Scalar tensor view
    EXPECT_NEAR(view.at<float>(0), 7.0f, 1e-5);
}

TEST(TensorSubViewTest, ExpectValidSubViewAccessUsingVectorOfIndices) {
    // 2x3 tensor
    Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Access a sub-view using a vector of indices
    Tensor view = a[{1, 2}];  // Accessing the element at (1,2)

    ValidateTensorShape(view, 1, 1, {1});  // Scalar tensor view
    EXPECT_NEAR(view.at<float>(0), 6.0f, 1e-5);
}


TEST(TensorSubViewTest, ExpectSubViewModificationUsingInitializerList) {
    // 2x2 tensor
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor view = a[{1, 1}];

    view += 10.0f;

    // Check that the modification is reflected in the original tensor
    ValidateTensorData(a, std::vector<float>{1.0f, 2.0f, 3.0f, 14.0f});
}

/*

TEST OPERATIONS WITH PROMOTION (+, -, *, /)

*/

TEST(TensorBroadcastingTest, ExpectValidAdditionWithPromotion) {
    std::vector<int> dataInt = {1, 2, 3, 4, 5, 6};
    Tensor a({2, 3}, dataInt);
    std::vector<float> dataFloat = {1.5f, 2.5f, 3.5f};
    Tensor b({3}, dataFloat);

    std::cout << dtypeToString(a.type()) << " - int" << std::endl;
    std::cout << dtypeToString(b.type()) << " - float" << std::endl;

    Tensor c = b + a;

    std::cout << c.toString() << std::endl;

    ValidateTensorShape(c, 2, 6, {2, 3});
    ValidateTensorData(c, std::vector<float>{2.5f, 4.5f, 6.5f, 5.5f, 7.5f, 9.5f});
    std::cout << dtypeToString(c.type()) << std::endl;
    std::cout << dtypeToString(dtype::f32) << std::endl;
    EXPECT_EQ(c.type(), dtype::f64);
}

} // namespace sdnn

#endif // TEST_TENSOR_CPP
#ifndef TEST_TEMPLATE_TENSOR_CPP
#define TEST_TEMPLATE_TENSOR_CPP

#include "../utils/tensor_helpers.hpp"

namespace smart_dnn {

/*
 * TEMPLATE TENSOR TYPE TESTS
 * Testing tensor functionality with different data types (float, double, int)
 */

// ============================================================================
// FLOAT TENSOR TESTS
// ============================================================================

TEST(TemplateTensorFloatTest, InitializationWithFloatType) {
    Tensor<float> a({2, 3}, 5.5f);
    
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateTensorData(a, std::vector<float>(6, 5.5f));
}

TEST(TemplateTensorFloatTest, FloatTensorArithmetic) {
    Tensor<float> a({2, 2}, 2.5f);
    Tensor<float> b({2, 2}, 1.5f);
    
    Tensor<float> sum = a + b;
    Tensor<float> diff = a - b;
    Tensor<float> prod = a * b;
    Tensor<float> quot = a / b;
    
    ValidateTensorData(sum, std::vector<float>(4, 4.0f));
    ValidateTensorData(diff, std::vector<float>(4, 1.0f));
    ValidateTensorData(prod, std::vector<float>(4, 3.75f));
    
    std::vector<float> expected_quot(4, 2.5f / 1.5f);
    ValidateTensorData(quot, expected_quot);
}

TEST(TemplateTensorFloatTest, FloatTensorScalarOperations) {
    Tensor<float> a({2, 2}, 10.0f);
    
    Tensor<float> add = a + 5.0f;
    Tensor<float> sub = a - 3.0f;
    Tensor<float> mul = a * 2.0f;
    Tensor<float> div = a / 2.0f;
    
    ValidateTensorData(add, std::vector<float>(4, 15.0f));
    ValidateTensorData(sub, std::vector<float>(4, 7.0f));
    ValidateTensorData(mul, std::vector<float>(4, 20.0f));
    ValidateTensorData(div, std::vector<float>(4, 5.0f));
}

// ============================================================================
// DOUBLE TENSOR TESTS
// ============================================================================

TEST(TemplateTensorDoubleTest, InitializationWithDoubleType) {
    Tensor<double> a({2, 3}, 5.5);
    
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateTensorData(a, std::vector<double>(6, 5.5));
}

TEST(TemplateTensorDoubleTest, DoubleTensorArithmetic) {
    Tensor<double> a({2, 2}, 2.5);
    Tensor<double> b({2, 2}, 1.5);
    
    Tensor<double> sum = a + b;
    Tensor<double> diff = a - b;
    Tensor<double> prod = a * b;
    Tensor<double> quot = a / b;
    
    ValidateTensorData(sum, std::vector<double>(4, 4.0));
    ValidateTensorData(diff, std::vector<double>(4, 1.0));
    ValidateTensorData(prod, std::vector<double>(4, 3.75));
    
    std::vector<double> expected_quot(4, 2.5 / 1.5);
    ValidateTensorData(quot, expected_quot);
}

TEST(TemplateTensorDoubleTest, DoubleTensorScalarOperations) {
    Tensor<double> a({2, 2}, 10.0);
    
    Tensor<double> add = a + 5.0;
    Tensor<double> sub = a - 3.0;
    Tensor<double> mul = a * 2.0;
    Tensor<double> div = a / 2.0;
    
    ValidateTensorData(add, std::vector<double>(4, 15.0));
    ValidateTensorData(sub, std::vector<double>(4, 7.0));
    ValidateTensorData(mul, std::vector<double>(4, 20.0));
    ValidateTensorData(div, std::vector<double>(4, 5.0));
}

TEST(TemplateTensorDoubleTest, DoubleTensorPrecision) {
    // Test that double provides higher precision than float
    Tensor<double> a({1, 1}, 1.0 / 3.0);
    Tensor<double> b({1, 1}, 3.0);
    Tensor<double> result = a * b;
    
    // Should be very close to 1.0 with double precision
    ASSERT_NEAR(result.getData()[0], 1.0, 1e-15);
}

// ============================================================================
// INTEGER TENSOR TESTS
// ============================================================================

TEST(TemplateTensorIntTest, InitializationWithIntType) {
    Tensor<int> a({2, 3}, 5);
    
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateTensorData(a, std::vector<int>(6, 5));
}

TEST(TemplateTensorIntTest, IntTensorArithmetic) {
    Tensor<int> a({2, 2}, 10);
    Tensor<int> b({2, 2}, 3);
    
    Tensor<int> sum = a + b;
    Tensor<int> diff = a - b;
    Tensor<int> prod = a * b;
    Tensor<int> quot = a / b;
    
    ValidateTensorData(sum, std::vector<int>(4, 13));
    ValidateTensorData(diff, std::vector<int>(4, 7));
    ValidateTensorData(prod, std::vector<int>(4, 30));
    ValidateTensorData(quot, std::vector<int>(4, 3));  // Integer division
}

TEST(TemplateTensorIntTest, IntTensorScalarOperations) {
    Tensor<int> a({2, 2}, 10);
    
    Tensor<int> add = a + 5;
    Tensor<int> sub = a - 3;
    Tensor<int> mul = a * 2;
    Tensor<int> div = a / 2;
    
    ValidateTensorData(add, std::vector<int>(4, 15));
    ValidateTensorData(sub, std::vector<int>(4, 7));
    ValidateTensorData(mul, std::vector<int>(4, 20));
    ValidateTensorData(div, std::vector<int>(4, 5));
}

// ============================================================================
// COPY AND MOVE TESTS FOR TEMPLATE TYPES
// ============================================================================

TEST(TemplateTensorCopyTest, DoubleTensorCopy) {
    std::vector<double> data = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    Tensor<double> a({2, 3}, data);
    Tensor<double> b(a);
    
    ValidateTensorShape(b, 2, 6, {2, 3});
    ValidateTensorData(b, data);
}

TEST(TemplateTensorCopyTest, IntTensorCopy) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    Tensor<int> a({2, 3}, data);
    Tensor<int> b(a);
    
    ValidateTensorShape(b, 2, 6, {2, 3});
    ValidateTensorData(b, data);
}

TEST(TemplateTensorMoveTest, DoubleTensorMove) {
    std::vector<double> data = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    Tensor<double> a({2, 3}, data);
    Tensor<double> b(std::move(a));
    
    ValidateTensorShape(b, 2, 6, {2, 3});
    ValidateTensorData(b, data);
}

TEST(TemplateTensorMoveTest, IntTensorMove) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    Tensor<int> a({2, 3}, data);
    Tensor<int> b(std::move(a));
    
    ValidateTensorShape(b, 2, 6, {2, 3});
    ValidateTensorData(b, data);
}

// ============================================================================
// ASSIGNMENT OPERATORS FOR TEMPLATE TYPES
// ============================================================================

TEST(TemplateTensorAssignmentTest, DoubleTensorAssignment) {
    Tensor<double> a({2, 2}, 5.5);
    Tensor<double> b({2, 2}, 3.3);
    
    a += b;
    ValidateTensorData(a, std::vector<double>(4, 8.8));
}

TEST(TemplateTensorAssignmentTest, IntTensorAssignment) {
    Tensor<int> a({2, 2}, 10);
    Tensor<int> b({2, 2}, 5);
    
    a -= b;
    ValidateTensorData(a, std::vector<int>(4, 5));
}

// ============================================================================
// VECTOR INITIALIZATION TESTS
// ============================================================================

TEST(TemplateTensorVectorInitTest, FloatVectorInit) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<float> a({2, 2}, data);
    
    ValidateTensorShape(a, 2, 4, {2, 2});
    ValidateTensorData(a, data);
}

TEST(TemplateTensorVectorInitTest, DoubleVectorInit) {
    std::vector<double> data = {1.1, 2.2, 3.3, 4.4};
    Tensor<double> a({2, 2}, data);
    
    ValidateTensorShape(a, 2, 4, {2, 2});
    ValidateTensorData(a, data);
}

TEST(TemplateTensorVectorInitTest, IntVectorInit) {
    std::vector<int> data = {1, 2, 3, 4};
    Tensor<int> a({2, 2}, data);
    
    ValidateTensorShape(a, 2, 4, {2, 2});
    ValidateTensorData(a, data);
}

// ============================================================================
// FACTORY METHOD TESTS FOR TEMPLATE TYPES
// ============================================================================

TEST(TemplateTensorFactoryTest, FloatOnes) {
    Tensor<float> a = Tensor<float>::ones({2, 3});
    
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateTensorData(a, std::vector<float>(6, 1.0f));
}

TEST(TemplateTensorFactoryTest, DoubleZeros) {
    Tensor<double> a = Tensor<double>::zeros({2, 3});
    
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateTensorData(a, std::vector<double>(6, 0.0));
}

TEST(TemplateTensorFactoryTest, IntOnes) {
    Tensor<int> a = Tensor<int>::ones({2, 3});
    
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateTensorData(a, std::vector<int>(6, 1));
}

TEST(TemplateTensorFactoryTest, IntZeros) {
    Tensor<int> a = Tensor<int>::zeros({2, 3});
    
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateTensorData(a, std::vector<int>(6, 0));
}

TEST(TemplateTensorFactoryTest, FloatRandom) {
    Tensor<float> a = Tensor<float>::rand({2, 3});
    
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateRandomTensor(a, 0.0f, 1.0f);
}

TEST(TemplateTensorFactoryTest, DoubleRandom) {
    Tensor<double> a = Tensor<double>::rand({2, 3});
    
    ValidateTensorShape(a, 2, 6, {2, 3});
    ValidateRandomTensor(a, 0.0, 1.0);
}

// ============================================================================
// ELEMENT ACCESS TESTS FOR TEMPLATE TYPES
// ============================================================================

TEST(TemplateTensorAccessTest, DoubleElementAccess) {
    std::vector<double> data = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    Tensor<double> a({2, 3}, data);
    
    ASSERT_NEAR(a[0], 1.1, 1e-10);
    ASSERT_NEAR(a[3], 4.4, 1e-10);
    ASSERT_NEAR(a[5], 6.6, 1e-10);
}

TEST(TemplateTensorAccessTest, IntElementAccess) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    Tensor<int> a({2, 3}, data);
    
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(a[3], 4);
    ASSERT_EQ(a[5], 6);
}

TEST(TemplateTensorAccessTest, FloatElementModification) {
    Tensor<float> a({2, 2}, 5.0f);
    
    a[0] = 10.0f;
    a[3] = 15.0f;
    
    ASSERT_FLOAT_EQ(a[0], 10.0f);
    ASSERT_FLOAT_EQ(a[3], 15.0f);
}

// ============================================================================
// BROADCASTING TESTS FOR TEMPLATE TYPES
// ============================================================================

TEST(TemplateTensorBroadcastTest, DoubleBroadcast) {
    Tensor<double> a({2, 3}, 5.0);
    Tensor<double> b({1, 3}, 2.0);
    
    Tensor<double> result = a + b;
    
    ValidateTensorShape(result, 2, 6, {2, 3});
    ValidateTensorData(result, std::vector<double>(6, 7.0));
}

TEST(TemplateTensorBroadcastTest, IntBroadcast) {
    Tensor<int> a({2, 3}, 10);
    Tensor<int> b({1, 3}, 5);
    
    Tensor<int> result = a - b;
    
    ValidateTensorShape(result, 2, 6, {2, 3});
    ValidateTensorData(result, std::vector<int>(6, 5));
}

// ============================================================================
// NEGATIVE OPERATION TESTS FOR TEMPLATE TYPES
// ============================================================================

TEST(TemplateTensorNegativeTest, FloatNegation) {
    Tensor<float> a({2, 2}, 5.0f);
    Tensor<float> b = -a;
    
    ValidateTensorData(b, std::vector<float>(4, -5.0f));
}

TEST(TemplateTensorNegativeTest, DoubleNegation) {
    Tensor<double> a({2, 2}, 3.5);
    Tensor<double> b = -a;
    
    ValidateTensorData(b, std::vector<double>(4, -3.5));
}

TEST(TemplateTensorNegativeTest, IntNegation) {
    Tensor<int> a({2, 2}, 7);
    Tensor<int> b = -a;
    
    ValidateTensorData(b, std::vector<int>(4, -7));
}

// ============================================================================
// EQUALITY TESTS FOR TEMPLATE TYPES
// ============================================================================

TEST(TemplateTensorEqualityTest, FloatEquality) {
    Tensor<float> a({2, 2}, 5.0f);
    Tensor<float> b({2, 2}, 5.0f);
    Tensor<float> c({2, 2}, 3.0f);
    
    ASSERT_TRUE(a == b);
    ASSERT_FALSE(a == c);
    ASSERT_TRUE(a != c);
}

TEST(TemplateTensorEqualityTest, DoubleEquality) {
    Tensor<double> a({2, 2}, 5.5);
    Tensor<double> b({2, 2}, 5.5);
    Tensor<double> c({2, 2}, 3.3);
    
    ASSERT_TRUE(a == b);
    ASSERT_FALSE(a == c);
    ASSERT_TRUE(a != c);
}

TEST(TemplateTensorEqualityTest, IntEquality) {
    Tensor<int> a({2, 2}, 5);
    Tensor<int> b({2, 2}, 5);
    Tensor<int> c({2, 2}, 3);
    
    ASSERT_TRUE(a == b);
    ASSERT_FALSE(a == c);
    ASSERT_TRUE(a != c);
}

} // namespace smart_dnn

#endif // TEST_TEMPLATE_TENSOR_CPP

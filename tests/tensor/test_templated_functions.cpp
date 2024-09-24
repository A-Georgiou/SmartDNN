#include <gtest/gtest.h>
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/Backend/Default/TemplatedOperations.hpp"

namespace sdnn {

class TemplatedFunctionTests : public ::testing::Test {
protected:
    Tensor createTensor(const std::vector<int>& shape_, const std::vector<float>& values) {
        return Tensor(Shape(shape_), values);
    }
};

/*

    TEST APPLY OPERATION FUNCTION

*/

TEST_F(TemplatedFunctionTests, ApplyOperationTest) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    // Apply square operation
    Tensor result = applyOperation(a, [](float x) { return x * x; });

    std::vector<float> expected = {1.0f, 4.0f, 9.0f, 16.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TemplatedFunctionTests, ApplyOperationNegativeTest) {
    Tensor a = createTensor({2, 2}, {-1.0f, -2.0f, -3.0f, -4.0f});

    // Apply abs operation
    Tensor result = applyOperation(a, [](float x) { return std::abs(x); });

    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

/*

    TEST ELEMENT-WISE OPERATION FUNCTION

*/

TEST_F(TemplatedFunctionTests, ElementWiseOpTest) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

    // Apply element-wise addition
    Tensor result = elementWiseOp(a, b, [](float x, float y) { return x + y; });

    std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TemplatedFunctionTests, ElementWiseOpMismatchShapeTest) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = createTensor({2, 3}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});

    // Expect an exception due to shape mismatch
    ASSERT_THROW(elementWiseOp(a, b, [](float x, float y) { return x + y; }), std::invalid_argument);
}

/*

    TEST SCALAR OPERATION FUNCTION

*/

TEST_F(TemplatedFunctionTests, ScalarOpTest) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    double scalar = 2.0;

    // Apply scalar multiplication
    Tensor result = scalarOp(a, scalar, [](float x, float y) { return x * y; });

    std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TemplatedFunctionTests, ScalarOpDivisionTest) {
    Tensor a = createTensor({2, 2}, {4.0f, 8.0f, 12.0f, 16.0f});
    double scalar = 4.0;

    // Apply scalar division
    Tensor result = scalarOp(a, scalar, [](float x, float y) { return x / y; });

    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

/*

    TEST REDUCTION FUNCTION

*/

TEST_F(TemplatedFunctionTests, ReductionSumTest) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    float defaultValue = 0.0f;

    // Perform sum reduction over axis 0
    Tensor result = reduction(a, {0}, false, [](float a, float b) { return a + b; }, [](float sum, size_t) { return sum; }, DataItem{&defaultValue, dtype::f32});
    std::vector<float> expected = {4.0f, 6.0f};  // Sum across axis 0
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TemplatedFunctionTests, ReductionMeanTest) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    float defaultValue = 0.0f;

    // Perform mean reduction over axis 0
    Tensor result = reduction(a, {0}, false, [](float a, float b) { return a + b; }, [](float sum, size_t count) { return sum / count; }, DataItem{&defaultValue, dtype::f32});
    std::vector<float> expected = {2.0f, 3.0f};  // Mean across axis 0
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TemplatedFunctionTests, ReductionWithKeepDimsTest) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    float defaultValue = 0.0f;

    // Perform sum reduction over axis 0, keeping dimensions
    Tensor result = reduction(a, {0}, true, [](float a, float b) { return a + b; }, [](float sum, size_t) { return sum; }, DataItem{&defaultValue, dtype::f32});

    EXPECT_EQ(result.shape(), Shape({1, 2}));  // Keep dimensions

    std::vector<float> expected = {4.0f, 6.0f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
    }
}

TEST_F(TemplatedFunctionTests, InvalidReductionAxisTest) {
    Tensor a = createTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    float defaultValue = 0.0f;

    // Expect an exception due to invalid axis
    ASSERT_THROW(reduction(a, {2}, false, [](float a, float b) { return a + b; }, [](float sum, size_t) { return sum; }, DataItem{&defaultValue, dtype::f32}), std::invalid_argument);
}


} // namespace sdnn
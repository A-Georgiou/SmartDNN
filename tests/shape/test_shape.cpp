#ifndef TEST_SHAPE_CPP
#define TEST_SHAPE_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Shape/Shape.hpp"
#include <sstream>

// ==================== Shape Construction Tests ====================

TEST(ShapeConstructionTest, InitializerListConstruction) {
    Shape shape({2, 3, 4});
    
    ASSERT_EQ(shape.rank(), 3);
    ASSERT_EQ(shape.size(), 24);
    ASSERT_EQ(shape[0], 2);
    ASSERT_EQ(shape[1], 3);
    ASSERT_EQ(shape[2], 4);
}

TEST(ShapeConstructionTest, VectorConstruction) {
    std::vector<int> dims = {3, 4, 5};
    Shape shape(dims);
    
    ASSERT_EQ(shape.rank(), 3);
    ASSERT_EQ(shape.size(), 60);
    ASSERT_EQ(shape[0], 3);
    ASSERT_EQ(shape[1], 4);
    ASSERT_EQ(shape[2], 5);
}

TEST(ShapeConstructionTest, SingleDimensionShape) {
    Shape shape({10});
    
    ASSERT_EQ(shape.rank(), 1);
    ASSERT_EQ(shape.size(), 10);
    ASSERT_EQ(shape[0], 10);
}

TEST(ShapeConstructionTest, LargeDimensionShape) {
    Shape shape({10, 20, 30, 40});
    
    ASSERT_EQ(shape.rank(), 4);
    ASSERT_EQ(shape.size(), 240000);
}

TEST(ShapeConstructionTest, CopyConstruction) {
    Shape original({2, 3, 4});
    Shape copy(original);
    
    ASSERT_EQ(copy.rank(), original.rank());
    ASSERT_EQ(copy.size(), original.size());
    ASSERT_EQ(copy, original);
}

TEST(ShapeConstructionTest, MoveConstruction) {
    Shape original({2, 3, 4});
    Shape moved(std::move(original));
    
    ASSERT_EQ(moved.rank(), 3);
    ASSERT_EQ(moved.size(), 24);
    ASSERT_EQ(moved[0], 2);
    ASSERT_EQ(moved[1], 3);
    ASSERT_EQ(moved[2], 4);
}

// ==================== Shape Validation Tests ====================

TEST(ShapeValidationTest, EmptyDimensionsThrows) {
    EXPECT_THROW({
        Shape shape({});
    }, std::invalid_argument);
}

TEST(ShapeValidationTest, ZeroDimensionThrows) {
    EXPECT_THROW({
        Shape shape({2, 0, 3});
    }, std::invalid_argument);
}

TEST(ShapeValidationTest, NegativeDimensionThrows) {
    EXPECT_THROW({
        Shape shape({2, -3, 4});
    }, std::invalid_argument);
}

TEST(ShapeValidationTest, AllNegativeDimensionsThrows) {
    EXPECT_THROW({
        Shape shape({-1, -2, -3});
    }, std::invalid_argument);
}

// ==================== Shape Assignment Tests ====================

TEST(ShapeAssignmentTest, CopyAssignment) {
    Shape original({2, 3, 4});
    Shape assigned({1, 1});
    
    assigned = original;
    
    ASSERT_EQ(assigned.rank(), original.rank());
    ASSERT_EQ(assigned.size(), original.size());
    ASSERT_EQ(assigned, original);
}

TEST(ShapeAssignmentTest, MoveAssignment) {
    Shape original({2, 3, 4});
    Shape assigned({1, 1});
    
    assigned = std::move(original);
    
    ASSERT_EQ(assigned.rank(), 3);
    ASSERT_EQ(assigned.size(), 24);
    ASSERT_EQ(assigned[0], 2);
}

TEST(ShapeAssignmentTest, SelfAssignment) {
    Shape shape({2, 3, 4});
    shape = shape;
    
    ASSERT_EQ(shape.rank(), 3);
    ASSERT_EQ(shape.size(), 24);
}

// ==================== Shape Reshape Tests ====================

TEST(ShapeReshapeTest, ReshapeWithSameSize) {
    Shape shape({2, 3, 4});
    shape.reshape(std::vector<int>{6, 4});
    
    ASSERT_EQ(shape.rank(), 2);
    ASSERT_EQ(shape.size(), 24);
    ASSERT_EQ(shape[0], 6);
    ASSERT_EQ(shape[1], 4);
}

TEST(ShapeReshapeTest, ReshapeToSingleDimension) {
    Shape shape({2, 3, 4});
    shape.reshape(std::vector<int>{24});
    
    ASSERT_EQ(shape.rank(), 1);
    ASSERT_EQ(shape.size(), 24);
    ASSERT_EQ(shape[0], 24);
}

TEST(ShapeReshapeTest, ReshapeToHigherDimensions) {
    Shape shape({24});
    shape.reshape(std::vector<int>{2, 3, 4});
    
    ASSERT_EQ(shape.rank(), 3);
    ASSERT_EQ(shape.size(), 24);
}

TEST(ShapeReshapeTest, ReshapeWithShapeObject) {
    Shape shape({2, 3, 4});
    Shape newShape({6, 4});
    
    shape.reshape(newShape);
    
    ASSERT_EQ(shape.rank(), 2);
    ASSERT_EQ(shape[0], 6);
    ASSERT_EQ(shape[1], 4);
}

TEST(ShapeReshapeTest, ReshapeWithMismatchedSizeThrows) {
    Shape shape({2, 3, 4});
    
    EXPECT_THROW({
        shape.reshape(std::vector<int>{5, 5});  // Size 25 != 24
    }, std::runtime_error);
}

TEST(ShapeReshapeTest, ReshapeWithInvalidDimensionsThrows) {
    Shape shape({2, 3, 4});
    
    EXPECT_THROW({
        shape.reshape(std::vector<int>{0, 24});
    }, std::invalid_argument);
}

// ==================== Shape Comparison Tests ====================

TEST(ShapeComparisonTest, EqualityOperator) {
    Shape shape1({2, 3, 4});
    Shape shape2({2, 3, 4});
    Shape shape3({2, 3, 5});
    
    ASSERT_TRUE(shape1 == shape2);
    ASSERT_FALSE(shape1 == shape3);
}

TEST(ShapeComparisonTest, InequalityOperator) {
    Shape shape1({2, 3, 4});
    Shape shape2({2, 3, 5});
    Shape shape3({2, 3, 4});
    
    ASSERT_TRUE(shape1 != shape2);
    ASSERT_FALSE(shape1 != shape3);
}

TEST(ShapeComparisonTest, DifferentRanksAreNotEqual) {
    Shape shape1({2, 3});
    Shape shape2({2, 3, 1});
    
    ASSERT_TRUE(shape1 != shape2);
    ASSERT_FALSE(shape1 == shape2);
}

// ==================== Shape Access Tests ====================

TEST(ShapeAccessTest, IndexOperatorRead) {
    Shape shape({2, 3, 4, 5});
    
    ASSERT_EQ(shape[0], 2);
    ASSERT_EQ(shape[1], 3);
    ASSERT_EQ(shape[2], 4);
    ASSERT_EQ(shape[3], 5);
}

TEST(ShapeAccessTest, IndexOperatorWrite) {
    Shape shape({2, 3, 4});
    
    shape[1] = 5;
    
    ASSERT_EQ(shape[1], 5);
}

TEST(ShapeAccessTest, GetDimensions) {
    Shape shape({2, 3, 4});
    const std::vector<int>& dims = shape.getDimensions();
    
    ASSERT_EQ(dims.size(), 3);
    ASSERT_EQ(dims[0], 2);
    ASSERT_EQ(dims[1], 3);
    ASSERT_EQ(dims[2], 4);
}

TEST(ShapeAccessTest, GetStride) {
    Shape shape({2, 3, 4});
    const std::vector<size_t>& stride = shape.getStride();
    
    // For shape (2, 3, 4), strides should be [12, 4, 1]
    ASSERT_EQ(stride.size(), 3);
    ASSERT_EQ(stride[0], 12);
    ASSERT_EQ(stride[1], 4);
    ASSERT_EQ(stride[2], 1);
}

TEST(ShapeAccessTest, GetStrideForSingleDimension) {
    Shape shape({10});
    const std::vector<size_t>& stride = shape.getStride();
    
    ASSERT_EQ(stride.size(), 1);
    ASSERT_EQ(stride[0], 1);
}

// ==================== Shape Iterator Tests ====================

TEST(ShapeIteratorTest, BeginEndIterators) {
    Shape shape({2, 3, 4});
    
    auto it = shape.begin();
    ASSERT_EQ(*it, 2);
    ++it;
    ASSERT_EQ(*it, 3);
    ++it;
    ASSERT_EQ(*it, 4);
    ++it;
    ASSERT_EQ(it, shape.end());
}

TEST(ShapeIteratorTest, RangeBasedForLoop) {
    Shape shape({2, 3, 4});
    std::vector<int> dims;
    
    for (int dim : shape) {
        dims.push_back(dim);
    }
    
    ASSERT_EQ(dims.size(), 3);
    ASSERT_EQ(dims[0], 2);
    ASSERT_EQ(dims[1], 3);
    ASSERT_EQ(dims[2], 4);
}

// ==================== Shape String Representation Tests ====================

TEST(ShapeStringTest, ToStringMethod) {
    Shape shape({2, 3, 4});
    std::string str = shape.toString();
    
    ASSERT_EQ(str, "(2, 3, 4)");
}

TEST(ShapeStringTest, ToStringForSingleDimension) {
    Shape shape({10});
    std::string str = shape.toString();
    
    ASSERT_EQ(str, "(10)");
}

TEST(ShapeStringTest, StreamOutputOperator) {
    Shape shape({2, 3, 4});
    std::ostringstream oss;
    
    oss << shape;
    
    ASSERT_EQ(oss.str(), "(2, 3, 4)");
}

// ==================== Shape Size Calculation Tests ====================

TEST(ShapeSizeTest, Size1D) {
    Shape shape({10});
    ASSERT_EQ(shape.size(), 10);
}

TEST(ShapeSizeTest, Size2D) {
    Shape shape({3, 4});
    ASSERT_EQ(shape.size(), 12);
}

TEST(ShapeSizeTest, Size3D) {
    Shape shape({2, 3, 4});
    ASSERT_EQ(shape.size(), 24);
}

TEST(ShapeSizeTest, Size4D) {
    Shape shape({2, 3, 4, 5});
    ASSERT_EQ(shape.size(), 120);
}

TEST(ShapeSizeTest, LargeSize) {
    Shape shape({100, 100, 10});
    ASSERT_EQ(shape.size(), 100000);
}

// ==================== Shape Rank Tests ====================

TEST(ShapeRankTest, Rank1D) {
    Shape shape({10});
    ASSERT_EQ(shape.rank(), 1);
}

TEST(ShapeRankTest, Rank2D) {
    Shape shape({3, 4});
    ASSERT_EQ(shape.rank(), 2);
}

TEST(ShapeRankTest, Rank3D) {
    Shape shape({2, 3, 4});
    ASSERT_EQ(shape.rank(), 3);
}

TEST(ShapeRankTest, Rank5D) {
    Shape shape({1, 2, 3, 4, 5});
    ASSERT_EQ(shape.rank(), 5);
}

#endif // TEST_SHAPE_CPP

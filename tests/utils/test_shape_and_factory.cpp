#ifndef TEST_SHAPE_CPP
#define TEST_SHAPE_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/Shape/Shape.hpp"
#include "../../smart_dnn/Shape/ShapeOperations.hpp"
#include "tensor_helpers.hpp"

namespace smart_dnn {

// ============================================================================
// Shape Tests
// ============================================================================

TEST(ShapeTest, Construction) {
    Shape shape({2, 3, 4});
    
    EXPECT_EQ(shape.rank(), 3);
    EXPECT_EQ(shape.size(), 24);  // 2*3*4
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

TEST(ShapeTest, ConstructionFromVector) {
    std::vector<int> dims = {5, 6, 7};
    Shape shape(dims);
    
    EXPECT_EQ(shape.rank(), 3);
    EXPECT_EQ(shape.size(), 210);  // 5*6*7
}

TEST(ShapeTest, CopyConstructor) {
    Shape original({2, 3, 4});
    Shape copy(original);
    
    EXPECT_EQ(copy.rank(), original.rank());
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy[0], original[0]);
    EXPECT_EQ(copy[1], original[1]);
    EXPECT_EQ(copy[2], original[2]);
}

TEST(ShapeTest, MoveConstructor) {
    Shape original({2, 3, 4});
    size_t expectedSize = original.size();
    Shape moved(std::move(original));
    
    EXPECT_EQ(moved.rank(), 3);
    EXPECT_EQ(moved.size(), expectedSize);
}

TEST(ShapeTest, CopyAssignment) {
    Shape shape1({2, 3});
    Shape shape2({4, 5, 6});
    
    shape1 = shape2;
    
    EXPECT_EQ(shape1.rank(), 3);
    EXPECT_EQ(shape1.size(), 120);
    EXPECT_EQ(shape1[0], 4);
}

TEST(ShapeTest, MoveAssignment) {
    Shape shape1({2, 3});
    Shape shape2({4, 5, 6});
    
    shape1 = std::move(shape2);
    
    EXPECT_EQ(shape1.rank(), 3);
    EXPECT_EQ(shape1.size(), 120);
}

TEST(ShapeTest, Equality) {
    Shape shape1({2, 3, 4});
    Shape shape2({2, 3, 4});
    Shape shape3({2, 3, 5});
    
    EXPECT_TRUE(shape1 == shape2);
    EXPECT_FALSE(shape1 == shape3);
}

TEST(ShapeTest, Inequality) {
    Shape shape1({2, 3, 4});
    Shape shape2({2, 3, 5});
    
    EXPECT_TRUE(shape1 != shape2);
}

TEST(ShapeTest, ToString) {
    Shape shape({2, 3, 4});
    std::string str = shape.toString();
    
    EXPECT_EQ(str, "(2, 3, 4)");
}

TEST(ShapeTest, InvalidDimensions) {
    // Negative dimensions should throw
    EXPECT_THROW(Shape({-1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(Shape({2, -3, 4}), std::invalid_argument);
}

TEST(ShapeTest, ZeroDimension) {
    // Zero dimension should throw (implementation doesn't allow it)
    EXPECT_THROW(Shape({0, 2, 3}), std::invalid_argument);
}

TEST(ShapeTest, GetDimensions) {
    Shape shape({2, 3, 4});
    const std::vector<int>& dims = shape.getDimensions();
    
    EXPECT_EQ(dims.size(), 3);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 3);
    EXPECT_EQ(dims[2], 4);
}

// ============================================================================
// ShapeOperations Tests
// ============================================================================

TEST(ShapeOperationsTest, Broadcast2D) {
    Shape shape1({1, 3});
    Shape shape2({2, 3});
    
    Shape result = ShapeOperations::broadcastShapes(shape1, shape2);
    
    EXPECT_EQ(result.rank(), 2);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
}

TEST(ShapeOperationsTest, BroadcastScalar) {
    Shape shape1({1});
    Shape shape2({2, 3, 4});
    
    Shape result = ShapeOperations::broadcastShapes(shape1, shape2);
    
    EXPECT_EQ(result.rank(), 3);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 4);
}

TEST(ShapeOperationsTest, AreBroadcastable) {
    Shape shape1({1, 3});
    Shape shape2({2, 3});
    
    EXPECT_TRUE(ShapeOperations::areBroadcastable(shape1, shape2));
}

TEST(ShapeOperationsTest, AreNotBroadcastable) {
    Shape shape1({2, 3});
    Shape shape2({2, 4});
    
    EXPECT_FALSE(ShapeOperations::areBroadcastable(shape1, shape2));
}

} // namespace smart_dnn

#endif // TEST_SHAPE_CPP

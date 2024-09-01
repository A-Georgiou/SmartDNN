#ifndef SHAPE_OPERATIONS_HPP
#define SHAPE_OPERATIONS_HPP

#include "Shape.hpp"

namespace smart_dnn {

class ShapeOperations {
public:
    ShapeOperations() = delete; // Prevent instantiation

    static Shape broadcastShapes(const Shape& shape1, const Shape& shape2) {
        std::vector<int> result;
        int size1 = shape1.rank();
        int size2 = shape2.rank();
        int maxSize = std::max(size1, size2);
        int minSize = std::min(size1, size2);

        for (int i = 0; i < maxSize; ++i) {
            if (i < minSize) {
                if (shape1[i] == shape2[i]) {
                    result.push_back(shape1[i]);
                } else if (shape1[i] == 1) {
                    result.push_back(shape2[i]);
                } else if (shape2[i] == 1) {
                    result.push_back(shape1[i]);
                } else {
                    throw std::runtime_error("Shapes are not broadcastable!");
                }
            } else {
                if (size1 > size2) {
                    result.push_back(shape1[i]);
                } else {
                    result.push_back(shape2[i]);
                }
            }
        }

        return Shape(result);
    }
};

} // namespace smart_dnn

#endif // SHAPE_OPERATIONS_HPP
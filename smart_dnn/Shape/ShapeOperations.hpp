#ifndef SHAPE_OPERATIONS_HPP
#define SHAPE_OPERATIONS_HPP

#include "smart_dnn/Shape/Shape.hpp"

namespace smart_dnn {

class ShapeOperations {
public:
    ShapeOperations() = delete; // Prevent instantiation

    static Shape broadcastShapes(const Shape& shape1, const Shape& shape2) {
        std::vector<int> result;
        int size1 = shape1.rank();
        int size2 = shape2.rank();
        int maxSize = std::max(size1, size2);

        for (int i = 0; i < maxSize; ++i) {
            int dim1 = (i < size1) ? shape1[size1 - 1 - i] : 1;
            int dim2 = (i < size2) ? shape2[size2 - 1 - i] : 1;

            if (dim1 == dim2) {
                result.push_back(dim1);
            } else if (dim1 == 1) {
                result.push_back(dim2);
            } else if (dim2 == 1) {
                result.push_back(dim1);
            } else {
                throw std::invalid_argument("Shapes are not broadcastable! Mismatch between shapes: " + shape1.toString() + " and " + shape2.toString());
            }
        }

        std::reverse(result.begin(), result.end());
        return Shape(result);
    }

    static bool areBroadcastable(const Shape& A, const Shape& B) {
        int lenA = A.rank();
        int lenB = B.rank();
        int maxLen = std::max(lenA, lenB);

        for (int i = 0; i < maxLen; ++i) {
            int dimA = (i < lenA) ? A[lenA - 1 - i] : 1;
            int dimB = (i < lenB) ? B[lenB - 1 - i] : 1;

            if (dimA != dimB && dimA != 1 && dimB != 1) {
                return false;
            }
        }

        return true;
    }

    static Shape concat(const Shape& shape1, const Shape& shape2) {
        std::vector<int> newDimensions = shape1.getDimensions();
        newDimensions.insert(newDimensions.end(), shape2.getDimensions().begin(), shape2.getDimensions().end());
        return Shape(newDimensions);
    }
    };

    // Accepts std::vector<int> or std::initializer_list<int>.
    template <typename Indices>
    static size_t computeFlatIndex(const Shape& shape, const Indices& indices) {
        if (indices.size() != shape.rank()) {
            throw std::invalid_argument("Number of indices must match the rank of the shape.");
        }

        const std::vector<size_t>& strides = shape.getStride();
        size_t flatIndex = 0;
        size_t i = 0;

        for (int index : indices) {
            if (index < 0 || index >= shape[i]) {
                throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
            }
            flatIndex += static_cast<size_t>(index) * strides[i++];
        }

        return flatIndex;
    }

    // Braced lists such as {n, c, h, w} cannot deduce the template above.
    static size_t computeFlatIndex(const Shape& shape, std::initializer_list<int> indices) {
        return computeFlatIndex<std::initializer_list<int>>(shape, indices);
    }

} // namespace smart_dnn

#endif // SHAPE_OPERATIONS_HPP
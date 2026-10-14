#ifndef SHAPE_OPERATIONS_HPP
#define SHAPE_OPERATIONS_HPP

#include "Shape.hpp"
#include <algorithm>

namespace sdnn {

class ShapeOperations {
public:
    ShapeOperations() = delete; // Prevent instantiation

    static inline Shape broadcastShapes(const Shape& shape1, const Shape& shape2) {
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

    static inline bool areBroadcastable(const Shape& A, const Shape& B) {
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

    static inline Shape concat(const Shape& shape1, const Shape& shape2) {
        std::vector<int> newDimensions = shape1.getDimensions();
        newDimensions.insert(newDimensions.end(), shape2.getDimensions().begin(), shape2.getDimensions().end());
        return Shape(newDimensions);
    }
    };

    static inline size_t computeFlatIndex(const Shape& shape, const std::vector<size_t>& indices) {
        if (indices.size() != shape.rank()) {
            throw std::invalid_argument("Number of indices must match the rank of the shape.");
        }

        const std::vector<size_t>& strides = shape.getStride();
        size_t flatIndex = 0;

        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || indices[i] >= static_cast<size_t>(shape[i])) {
                throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
            }
            flatIndex += static_cast<size_t>(indices[i]) * strides[i];
        }

        return flatIndex;
    }

    static inline std::vector<size_t> unflattenIndex(size_t flatIndex, const Shape& shape) {
        std::vector<size_t> indices(shape.rank());
        const std::vector<size_t>& strides = shape.getStride();

        if (flatIndex >= shape.size()) {
            throw std::out_of_range("Flat index out of bounds");
        }

        for (size_t i = 0; i < shape.rank(); ++i) {
            indices[i] = flatIndex / strides[i];
            flatIndex %= strides[i];
        }

        return indices;
    }


} // namespace smart_dnn

#endif // SHAPE_OPERATIONS_HPP
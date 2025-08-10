#ifndef TENSOR_INDEX_HPP
#define TENSOR_INDEX_HPP

#include <vector>
#include "smart_dnn/Shape/Shape.hpp"

namespace sdnn {

class Tensor;

class TensorIndex {
public:
    TensorIndex(const Shape& shape)
        : shape_(shape), strides_(calculateStrides(shape)), offset_(0) {}

    TensorIndex(const Shape& shape, std::vector<size_t> strides, size_t offset)
        : shape_(shape), strides_(std::move(strides)), offset_(offset) {}

    size_t flattenIndex(const std::vector<size_t>& indices) const {
        if (indices.size() > shape_.rank()) {
            throw std::invalid_argument("Too many indices for the tensor's rank. Expected at most " +
                                        std::to_string(shape_.rank()) + " indices, but got " + std::to_string(indices.size()));
        }

        size_t flatIndex = offset_;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= static_cast<size_t>(shape_[i])) {
                throw std::out_of_range("Index out of range at dimension " + std::to_string(i));
            }
            flatIndex += indices[i] * strides_[i];
        }
        return flatIndex;
    }

    TensorIndex slice(const std::vector<std::pair<size_t, size_t>>& ranges) const {
        if (ranges.size() != shape_.rank()) {
            throw std::invalid_argument("Invalid number of slice ranges");
        }
        
        std::vector<int> newShape;
        std::vector<size_t> newStrides = strides_;
        size_t newOffset = offset_;
        
        for (size_t i = 0; i < ranges.size(); ++i) {
            size_t start = ranges[i].first;
            size_t end = ranges[i].second;
            if (start >= static_cast<size_t>(shape_[i]) || end > static_cast<size_t>(shape_[i]) || start >= end) {
                throw std::out_of_range("Invalid slice range");
            }
            
            newShape.push_back(end - start);
            newOffset += start * strides_[i];
        }
        
        return TensorIndex(Shape(newShape), newStrides, newOffset);
    }

    const Shape& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t offset() const { return offset_; }

private:
    Shape shape_;
    std::vector<size_t> strides_;
    size_t offset_;

    static std::vector<size_t> calculateStrides(const Shape& shape) {
        std::vector<size_t> strides(shape.rank());
        size_t stride = 1;
        for (int i = shape.rank() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
};

}

#endif // TENSOR_INDEX_HPP
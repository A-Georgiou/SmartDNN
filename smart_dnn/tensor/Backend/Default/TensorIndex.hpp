#include <vector>
#include "smart_dnn/shape/Shape.hpp"

namespace sdnn {

class Tensor;

class TensorIndex {
public:
    TensorIndex(const Shape& shape)
        : shape_(shape), strides_(calculateStrides(shape)), offset_(0) {}

    TensorIndex(const Shape& shape, std::vector<size_t> strides, size_t offset)
        : shape_(shape), strides_(std::move(strides)), offset_(offset) {}

    size_t flattenIndex(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.rank()) {
            throw std::invalid_argument("Invalid number of indices");
        }
        
        size_t flatIndex = offset_;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
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
            if (start >= shape_[i] || end > shape_[i] || start >= end) {
                throw std::out_of_range("Invalid slice range");
            }
            
            newShape.push_back(end - start);
            newOffset += start * strides_[i];
        }
        
        return TensorIndex(Shape(newShape), newStrides, newOffset);
    }

    TensorIndex subIndex(size_t index) const {
        if (index >= shape_[0]) {
            throw std::out_of_range("Index out of range");
        }

        // If this is a scalar sub-view, return a new index with shape {1}
        std::vector<int> newShape = shape_.getDimensions();
        if (newShape.size() > 1) {
            newShape.erase(newShape.begin());  // Remove the first dimension
        } else {
            newShape = {1};  // Scalar shape
        }

        std::vector<size_t> newStrides = strides_;
        if (newStrides.size() > 1) {
            newStrides.erase(newStrides.begin());
        } else {
            newStrides = {1};  // Single-element stride
        }

        size_t newOffset = offset_ + index * strides_[0];

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
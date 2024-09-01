#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <iostream>
#include <numeric>
#include <vector>
#include <sstream>

struct Shape {
    ~Shape() = default;

    Shape(const Shape& other) noexcept
        : _dimensions(other._dimensions), _size(other._size), _stride(other._stride) {}

    Shape(Shape&& other) noexcept
        : _dimensions(std::move(other._dimensions)), _size(other._size), _stride(other._stride) {}

    explicit Shape(std::vector<int> dims) : _dimensions(std::move(dims)) {
        validateDimensions();
        _size = calculateSize();
        _stride = calculateStride();
    }

    Shape(std::initializer_list<int> dims) : _dimensions(dims) {
        validateDimensions();
        _size = calculateSize();
        _stride = calculateStride();
    }

    [[nodiscard]] int rank() const { return _dimensions.size(); }
    [[nodiscard]] size_t size() const { return _size; }
    
    friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << "(";
        for (int i = 0; i < shape._dimensions.size(); ++i) {
            os << shape._dimensions[i];
            if (i != shape._dimensions.size() - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }

    Shape& operator=(const Shape& other) {
        if (this != &other) {
            _dimensions = other._dimensions;
            _size = other._size;
        }
        return *this;
    }

    Shape& operator=(Shape&& other) noexcept {
        if (this != &other) {
            _dimensions = std::move(other._dimensions);
            _size = other._size;
        }
        return *this;
    }

    void setDimensions(const std::vector<int>& dims) {
        int newSize = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
        if (_size != newSize) {
            throw std::runtime_error("Shape size mismatch. Current size: " + std::to_string(_size) + ", New size: " + std::to_string(newSize));
        }
        _dimensions = dims;
        validateDimensions();
        _stride = calculateStride();
    }

    [[nodiscard]] std::vector<int> getDimensions() const { return _dimensions; }
    [[nodiscard]] std::vector<int> getStride() const { return _stride; }

    int operator[](int index) const { return _dimensions[index]; }
    bool operator==(const Shape& other) const { return _dimensions == other._dimensions; }
    bool operator!=(const Shape& other) const { return !(*this == other); }

private:
    int _size;
    std::vector<int> _dimensions;
    std::vector<int> _stride;

    int calculateSize() const {
        return std::accumulate(_dimensions.begin(), _dimensions.end(), 1, std::multiplies<int>());
    }

    std::vector<int> calculateStride() const {
        std::vector<int> stride(_dimensions.size(), 1);
        for (int i = _dimensions.size() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * _dimensions[i + 1];
        }
        return stride;
    }

    void validateDimensions() const {
        if (_dimensions.empty()) {
            throw std::invalid_argument("Shape must have at least one dimension");
        }
        for (int dim : _dimensions) {
            if (dim <= 0) {
                throw std::invalid_argument("Dimensions must be positive integers. Got: " + toString());
            }
        }
    }
};

#endif // SHAPE_HPP
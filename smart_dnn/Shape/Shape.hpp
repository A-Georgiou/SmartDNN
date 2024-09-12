#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <iostream>
#include <numeric>
#include <vector>
#include <sstream>

namespace sdnn {

struct Shape {
    ~Shape() = default;

    Shape(const Shape& other) noexcept
        : _dimensions(other._dimensions), _size(other._size), _stride(other._stride) {}

    Shape(Shape&& other) noexcept
        : _dimensions(std::move(other._dimensions)), _size(other._size), _stride(std::move(other._stride)) {}

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

    [[nodiscard]] size_t rank() const { return _dimensions.size(); }
    [[nodiscard]] size_t size() const { return _size; }
    
    friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << "(";
        for (size_t i = 0; i < shape._dimensions.size(); ++i) {
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
            _stride = other._stride;  
        }
        return *this;
    }

    Shape& operator=(Shape&& other) noexcept {
        if (this != &other) {
            _dimensions = std::move(other._dimensions);
            _size = other._size;
            _stride = std::move(other._stride);  
        }
        return *this;
    }

    void reshape(const Shape& other){
        reshape(other.getDimensions());
    }

    void reshape(const std::vector<int>& dims) {
        validateDimensions(dims);
        size_t newSize = 1;
        for (int dim : dims) {
            if (newSize > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
                throw std::overflow_error("New shape size exceeds maximum representable value");
            }
            newSize *= static_cast<size_t>(dim);
        }

        if (_size != newSize) {
            throw std::runtime_error("Shape size mismatch. Current size: " 
                + std::to_string(_size) + ", New size: " + std::to_string(newSize) 
                + ". New dimensions: " + Shape(dims).toString());
        }

        _dimensions = dims;
        _stride = calculateStride();
    }

    [[nodiscard]] const std::vector<int>& getDimensions() const { return _dimensions; }
    [[nodiscard]] const std::vector<size_t>& getStride() const { return _stride; }

    const int& operator[](size_t index) const { return _dimensions[index]; }
    int& operator[](size_t index) { return _dimensions[index]; }

    bool operator==(const Shape& other) const { return _dimensions == other._dimensions; }
    bool operator!=(const Shape& other) const { return !(*this == other); }

    auto begin() const { return _dimensions.begin(); }
    auto end() const { return _dimensions.end(); }

private:
    std::vector<int> _dimensions;
    size_t _size;
    std::vector<size_t> _stride;

    size_t calculateSize() const {
        return std::accumulate(_dimensions.begin(), _dimensions.end(), size_t(1), std::multiplies<size_t>());
    }

    std::vector<size_t> calculateStride() const {
        std::vector<size_t> stride(_dimensions.size(), 1);
        for (int i = static_cast<int>(_dimensions.size()) - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * static_cast<size_t>(_dimensions[i + 1]);
        }
        return stride;
    }


    void validateDimensions() const {
        validateDimensions(_dimensions);
    }

    void validateDimensions(const std::vector<int>& dims) const {
        if (dims.empty()) {
            throw std::invalid_argument("Shape must have at least one dimension");
        }
        for (int dim : dims) {
            if (dim <= 0) {
                throw std::invalid_argument("Dimensions must be positive integers. Got: " + toString());
            }
        }
    }
};

} // namespace sdnn

#endif // SHAPE_HPP
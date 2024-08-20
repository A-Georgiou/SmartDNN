
#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <iostream>
#include <numeric>
#include <vector>
#include <sstream>

/*
    Shape struct to represent the dimensions of a tensor.
*/
struct Shape {
    std::vector<int> dimensions;

    Shape() = default;
    ~Shape() = default;
    Shape(const Shape& other) = default;
    Shape(Shape&& other) noexcept = default;

    explicit Shape(std::vector<int> dims) : dimensions(std::move(dims)) {
        validateDimensions();
    }

    Shape(std::initializer_list<int> dims) : dimensions(dims) {
        validateDimensions();
    }

    [[nodiscard]] int rank() const { return dimensions.size(); }
    [[nodiscard]] int size() const { return std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int>()); }
    
    friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << "(";
        for (size_t i = 0; i < shape.dimensions.size(); ++i) {
            os << shape.dimensions[i];
            if (i != shape.dimensions.size() - 1) {
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

    Shape& operator=(const Shape& other) = default;
    Shape& operator=(Shape&& other) noexcept = default;

    int operator[](int index) const { return dimensions[index]; }
    bool operator==(const Shape& other) const { return dimensions == other.dimensions; }
    bool operator!=(const Shape& other) const { return !(*this == other); }

private:
    void validateDimensions() const {
        for (int dim : dimensions) {
            if (dim < 0) {
                throw std::invalid_argument("Dimensions must be non-negative integers. Got: " + toString());
            }
        }
    }
};

#endif // SHAPE_HPP
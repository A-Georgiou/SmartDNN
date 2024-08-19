#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <numeric>
#include <iostream>
#include <sstream>
#include "RandomEngine.hpp"

class TensorOperations;

/*
    Shape class to represent the dimensions of a tensor.
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
                throw std::invalid_argument("Dimensions must be non-negative integers.");
            }
        }
    }
};


/*
    Tensor class to represent a multi-dimensional array of floating-point numbers.
    Performs element-wise operations and supports basic linear algebra operations.
*/
class Tensor {
public:
    Tensor();
    Tensor(Shape dimensions);
    Tensor(Shape dimensions, float value);
    Tensor(Shape shape, std::vector<float> data);

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    ~Tensor() = default;

    float& operator()(std::initializer_list<int> indices);
    const float& operator()(std::initializer_list<int> indices) const;

    void swap(Tensor& other) noexcept;

    const Shape& shape() const { return _shape; }
    const std::vector<float>& getData() const { return data; }
    std::vector<float>& getData() { return data; }

    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    std::vector<int> size() const;
    int size(int axis) const;

    float sum() const;
    Tensor sqrt() const;
    Tensor sum(int axis) const;
    void transpose(int dim1, int dim2);
    void reshape(const Shape& newShape);
    Tensor reshape(const Shape& newShape) const;
    Tensor apply(std::function<float(float)> op) const;

    template<typename... Args>
    void reshape(Args... args) {
        _shape = Shape{args...};
        reshape(_shape);
    }

    void fill(float value);
    void randomize(float min, float max);
    void print() const;

    void toGPU();
    void toCPU();
    bool isOnGPU() const;

    // Add CUDA-based matrix operations
    void add(const Tensor& other);
    void subtract(const Tensor& other);

    friend class TensorOperations;

private:
    Shape _shape;
    std::vector<float> data;
    float* d_data; // Pointer to GPU memory
    bool onGPU;

    void allocateGPUMemory();
    void freeGPUMemory();
    void copyToGPU();
    void copyToCPU();

    std::vector<int> getBroadcastShape(const Tensor& other) const;
    std::vector<int> getBroadcastShape(const Shape& newShape) const;
    void checkCompatibility(const Tensor& other) const;
    void applyElementWiseOperation(const Tensor& other, std::function<float(float, float)> op, Tensor* result) const;
};

/*

Inverse operators for scalar-tensor operations.

*/

inline Tensor operator+(float scalar, const Tensor& tensor) {
    return tensor + scalar; 
}

inline Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

inline Tensor operator-(float scalar, const Tensor& tensor) {
    Tensor result(tensor.shape());
    for (size_t i = 0; i < tensor.getData().size(); ++i) {
        result.getData()[i] = scalar - tensor.getData()[i];
    }
    return result;
}

inline Tensor operator/(float scalar, const Tensor& tensor) {
    Tensor result(tensor.shape());  
    for (size_t i = 0; i < tensor.getData().size(); ++i) {
        result.getData()[i] = scalar / tensor.getData()[i];
    }
    return result;
}

#endif // TENSOR_HPP

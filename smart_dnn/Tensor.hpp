#ifndef TENSOR_H
#define TENSOR_H

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
    Shape(std::vector<int> dims) : dimensions(dims) {}
    Shape(std::initializer_list<int> dims) : dimensions(dims) {}
    Shape(const Shape& other) : dimensions(other.dimensions) {}
    Shape(Shape&& other) noexcept : dimensions(std::move(other.dimensions)) {}

    int rank() const { return dimensions.size(); }
    int size() const { return std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int>()); }
    std::string toString() const {
        std::ostringstream oss;
        oss << "(";
        for (size_t i = 0; i < dimensions.size(); ++i) {
            oss << dimensions[i];
            if (i != dimensions.size() - 1) {
                oss << ", ";
            }
        }
        oss << ")";
        return oss.str();
    }

    Shape operator=(const Shape& other) { dimensions = other.dimensions; return *this; }

    int operator[](int index) const { return dimensions[index]; }
    bool operator==(const Shape& other) const { return std::equal(dimensions.begin(), dimensions.end(), other.dimensions.begin()); }
    bool operator!=(const Shape& other) const { return !(*this == other); }
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
    Tensor(const std::vector<float>& data, const Shape& shape);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    virtual ~Tensor();

    float& operator()(std::initializer_list<int> indices);
    const float& operator()(std::initializer_list<int> indices) const;

    void swap(Tensor& other) noexcept;

    Shape shape() const { return _shape; }
    const std::vector<float>& getData() const { return data; }
    std::vector<float>& getData() { return data; }

    Tensor& operator=(Tensor other);
    Tensor& operator=(Tensor&& other) noexcept;

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

    int sum() const;
    Tensor sum(int axis) const;
    void transpose(int dim1, int dim2);
    void reshape(const Shape& newShape);

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
    void checkCompatibility(const Tensor& other) const;
    void applyElementWiseOperation(const Tensor& other, std::function<float(float, float)> op, Tensor* result) const;
};

#endif // TENSOR_H

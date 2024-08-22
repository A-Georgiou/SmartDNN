#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <numeric>
#include <iostream>
#include <sstream>
#include <functional>
#include "Shape.hpp"
#include "RandomEngine.hpp"

// Forward declaration for TensorOperations
class TensorOperations;

/**
 * Tensor class represents a multi-dimensional array of floating-point numbers.
 * It supports element-wise operations, linear algebra operations, and other tensor manipulations.
 */
class Tensor {

    #define MIN_PARALLEL_SIZE 1000

public:
    // Constructors
    Tensor() noexcept;
    explicit Tensor(Shape dimensions) noexcept;
    Tensor(Shape otherShape, float value) noexcept;
    Tensor(Shape otherShape, std::vector<float> data);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;

    // Destructor
    ~Tensor() = default;

    // Assignment operators
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // Element access
    float& operator()(std::initializer_list<int> indices);
    const float& operator()(std::initializer_list<int> indices) const;

    // Basic operations
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    Tensor& operator+=(float scalar) noexcept;
    Tensor& operator-=(float scalar) noexcept;
    Tensor& operator*=(float scalar) noexcept;
    Tensor& operator/=(float scalar) noexcept;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(float scalar) const noexcept;
    Tensor operator-(float scalar) const noexcept;
    Tensor operator*(float scalar) const noexcept;
    Tensor operator/(float scalar) const noexcept;

    // Shape and size
    const Shape& shape() const { return _shape; }
    inline std::vector<int> size() const noexcept;
    inline int size(int axis) const;

    // Tensor manipulations
    float sum () const;
    Tensor sum(int axis) const;
    Tensor sqrt() const;
    Tensor apply(std::function<float(float)> op) const;
    void transpose(int dim1, int dim2);
    void reshape(const Shape& newShape);
    Tensor reshape(const Shape& newShape) const;

    template<typename... Args>
    void reshape(Args... args) {
        _shape = Shape{args...};
        reshape(_shape);
    }

    // Initialization and data management
    void fill(float value) noexcept;
    void randomize(float min, float max);
    const std::vector<float>& getData() const { return data; }
    std::vector<float>& getData() { return data; }
    std::string toString() const;
    void print() const noexcept;

    // GPU support (to be implemented)
    void toGPU();
    void toCPU();
    bool isOnGPU() const;

    // Friends and helpers
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    friend class TensorOperations;

private:
    Shape _shape;
    std::vector<float> data;
    float* d_data = nullptr;
    bool onGPU = false;

    void allocateGPUMemory();
    void freeGPUMemory();
    void copyToGPU();
    void copyToCPU();

    void swap(Tensor& other) noexcept;

    std::vector<int> getBroadcastShape(const Tensor& other) const;
    inline std::vector<int> getBroadcastShape(const Shape& newShape) const;
    Tensor applyElementWiseOperation(const Tensor& other, std::function<float(float, float)> op) const;
    void applyElementWiseOperation(const Tensor& other, std::function<float(float, float)> op, Tensor* result) const;
};

// Scalar-Tensor operations
Tensor operator+(float scalar, const Tensor& tensor);
Tensor operator*(float scalar, const Tensor& tensor);
Tensor operator-(float scalar, const Tensor& tensor) noexcept;
Tensor operator/(float scalar, const Tensor& tensor) noexcept;

#endif // TENSOR_HPP

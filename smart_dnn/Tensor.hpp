#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <numeric>
#include <iostream>
#include <sstream>
#include "RandomEngine.hpp"
#include "Shape.hpp"

class TensorOperations;

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

    // Get the shape dimensions of the tensor.
    std::vector<int> size() const;

    // Get the size of the tensor along the specified axis.
    // Parameters:
    // axis: The axis to get the size of
    int size(int axis) const;

    // Basic toString implementation using osstringstream.
    std::string toString() const;

    // Compute the sum of the tensor across all axis.
    float sum() const;

    // Compute the sqrt of each value in the tensor.
    Tensor sqrt() const;

    // Compute the sum of the tensor along the specified axis.
    // Parameters:
    // axis: The axis to sum along
    Tensor sum(int axis) const;
    
    // Transpose the tensor along the specified dimensions.
    // Parameters:
    // dim1: The first dimension to transpose
    // dim2: The second dimension to transpose
    void transpose(int dim1, int dim2);

    // Reshape the tensor to the specified dimensions.
    // Parameters:
    // newShape: The dimensions of the new shape
    void reshape(const Shape& newShape);

    // Reshape the tensor to the specified dimensions.
    // Parameters:
    // newShape: The dimensions of the new shape
    Tensor reshape(const Shape& newShape) const;

    // Apply a function to each element of the tensor.
    // Parameters:
    // op: The function to apply
    Tensor apply(std::function<float(float)> op) const;

    // Reshape the tensor to the specified dimensions.
    // Parameters:
    // args: The dimensions of the new shape
    template<typename... Args>
    void reshape(Args... args) {
        _shape = Shape{args...};
        reshape(_shape);
    }

    // Fill the tensor with a specific value.
    // Parameters:
    // value: The value to fill the tensor with
    void fill(float value);

    // Fill the tensor with random values in the range [min, max].
    // Parameters:
    // min: The minimum value of the range
    // max: The maximum value of the range
    void randomize(float min, float max);

    // Print the tensor to the console.
    void print() const;

    // To-be-implemented: GPU support through CUDA API.
    void toGPU();
    void toCPU();
    bool isOnGPU() const;

    // Element-wise operations.
    // Parameters:
    // other: The tensor to perform the operation with
    void add(const Tensor& other);
    void subtract(const Tensor& other);

    friend class TensorOperations;
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

private:
    Shape _shape;
    std::vector<float> data;
    float* d_data; // Pointer to GPU memory
    bool onGPU;

    void allocateGPUMemory();
    void freeGPUMemory();
    void copyToGPU();
    void copyToCPU();

    void swap(Tensor& other) noexcept;

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

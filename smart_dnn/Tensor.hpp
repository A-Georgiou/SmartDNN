#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <numeric>
#include <iostream>

/*
    Shape class to represent the dimensions of a tensor.
*/
struct Shape {
    std::vector<int> dimensions;

    Shape() = default;
    Shape(std::initializer_list<int> dims) : dimensions(dims) {}

    int rank() const { return dimensions.size(); }
    int size() const { return std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int>()); }
    std::string toString() const {
        std::string result = "(";
        for (int i = 0; i < dimensions.size(); ++i) {
            std::cout << dimensions[i] << ", ";
        }
        result += ")";
        return result;
    }
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
    ~Tensor();

    void swap(Tensor& other) noexcept;

    Shape shape() const { return _shape; }

    Tensor& operator=(Tensor other);
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    Tensor& operator*=(const Tensor& other) = delete;
    Tensor& operator/=(const Tensor& other) = delete;
    Tensor operator*(const Tensor& other) const = delete;
    Tensor operator/(const Tensor& other) const = delete;

    void fill(float value);
    void randomize(float min, float max);
    void print() const;

    float& operator()(const std::vector<int>& indices);
    const float& operator()(const std::vector<int>& indices) const;

    void toGPU();
    void toCPU();
    bool isOnGPU() const;

    // Add CUDA-based matrix operations
    void add(const Tensor& other);
    void subtract(const Tensor& other);
    void multiply(const Tensor& other);

private:
    Shape _shape;
    std::vector<float> data;
    float* d_data; // Pointer to GPU memory
    bool onGPU;

    void allocateGPUMemory();
    void freeGPUMemory();
    void copyToGPU();
    void copyToCPU();
};

#endif // TENSOR_H

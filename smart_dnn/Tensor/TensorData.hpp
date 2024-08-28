#ifndef TENSOR_DATA_HPP
#define TENSOR_DATA_HPP

#include "Shape.hpp"

template <typename T>
class TensorData {
public:
    explicit TensorData(Shape dimensions) noexcept;
    TensorData(const Shape& dimensions, T value) noexcept;
    TensorData(const Shape& dimensions, const T[]& data);

    // Copy and move constructors
    TensorData(const TensorData& other);
    TensorData(TensorData&& other) noexcept;

    ~TensorData();

    T[]& getData() { return data; }
    const T[]& getData() const { return data; }

    void fill(T value);
    
    void allocateGPUMemory() = delete;
    void freeGPUMemory() = delete;
    void copyToGPU() = delete;
    void copyToCPU() = delete;

private:
    Shape _shape;
    T[] data; 
};

#endif // TENSOR_DATA_HPP
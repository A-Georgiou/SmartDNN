#ifndef TENSOR_DATA_HPP
#define TENSOR_DATA_HPP

#include <arrayfire.h> // Include ArrayFire for af::array support
#include "Shape.hpp"

template <typename T>
class TensorData {
public:
    explicit TensorData(Shape dimensions) noexcept;
    TensorData(Shape dimensions, T value) noexcept;
    TensorData(Shape dimensions, const af::array& data);
    // Copy and move constructors
    TensorData(const TensorData& other);
    TensorData(TensorData&& other) noexcept;
    ~TensorData() = default;

    af::array& getData() { return data; }
    const af::array& getData() const { return data; }

    void fill(T value);
    
    // GPU memory management is handled by ArrayFire, no need for manual management
    void allocateGPUMemory() = delete;
    void freeGPUMemory() = delete;
    void copyToGPU() = delete;
    void copyToCPU() = delete;

private:
    Shape _shape;
    af::array data;  // ArrayFire's array, which manages its own memory
};

#endif // TENSOR_DATA_HPP
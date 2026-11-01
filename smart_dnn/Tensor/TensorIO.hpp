#ifndef TENSOR_IO_HPP
#define TENSOR_IO_HPP

#include "smart_dnn/Tensor/Tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ios>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace smart_dnn {

// Values per 8-bit block: one float scale, then this many signed bytes (the layout llama.cpp calls Q8_0).
constexpr size_t int8BlockSize = 32;

// Per-stream switch set by SmartDNN::saveModel and loadModel; while on, tensor values are stored as 8-bit blocks.
inline long& int8TensorStorage(std::ios_base& stream) {
    static const int index = std::ios_base::xalloc();
    return stream.iword(index);
}

// Writes the tensor's rank, dimensions and values in native byte order.
template <typename T>
void saveTensor(std::ostream& os, const Tensor<T>& tensor) {
    const std::vector<int>& dims = tensor.getShape().getDimensions();
    const uint32_t rank = static_cast<uint32_t>(dims.size());
    os.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    os.write(reinterpret_cast<const char*>(dims.data()), static_cast<std::streamsize>(dims.size() * sizeof(int)));

    const T* values = tensor.getData().data();
    const size_t size = tensor.getShape().size();
    if (!int8TensorStorage(os)) {
        os.write(reinterpret_cast<const char*>(values), static_cast<std::streamsize>(size * sizeof(T)));
        return;
    }
    for (size_t start = 0; start < size; start += int8BlockSize) {
        const size_t count = std::min(int8BlockSize, size - start);
        float largest = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            largest = std::max(largest, std::abs(static_cast<float>(values[start + i])));
        }
        const float scale = largest / 127.0f;
        int8_t quantized[int8BlockSize] = {};
        for (size_t i = 0; i < count && scale > 0.0f; ++i) {
            quantized[i] = static_cast<int8_t>(std::lround(static_cast<float>(values[start + i]) / scale));
        }
        os.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
        os.write(reinterpret_cast<const char*>(quantized), static_cast<std::streamsize>(count));
    }
}

// Reads values written by saveTensor into a tensor that must already have the saved shape.
template <typename T>
void loadTensor(std::istream& is, Tensor<T>& tensor) {
    uint32_t rank = 0;
    is.read(reinterpret_cast<char*>(&rank), sizeof(rank));
    if (!is || rank != tensor.getShape().rank()) {
        throw std::runtime_error("Saved tensor rank does not match shape " + tensor.getShape().toString());
    }
    std::vector<int> dims(rank);
    is.read(reinterpret_cast<char*>(dims.data()), static_cast<std::streamsize>(rank * sizeof(int)));
    if (!is || dims != tensor.getShape().getDimensions()) {
        throw std::runtime_error("Saved tensor dimensions do not match shape " + tensor.getShape().toString());
    }

    T* values = tensor.getData().data();
    const size_t size = tensor.getShape().size();
    if (!int8TensorStorage(is)) {
        is.read(reinterpret_cast<char*>(values), static_cast<std::streamsize>(size * sizeof(T)));
    } else {
        for (size_t start = 0; start < size && is; start += int8BlockSize) {
            const size_t count = std::min(int8BlockSize, size - start);
            float scale = 0.0f;
            int8_t quantized[int8BlockSize] = {};
            is.read(reinterpret_cast<char*>(&scale), sizeof(scale));
            is.read(reinterpret_cast<char*>(quantized), static_cast<std::streamsize>(count));
            for (size_t i = 0; i < count; ++i) {
                values[start + i] = static_cast<T>(scale * static_cast<float>(quantized[i]));
            }
        }
    }
    if (!is) {
        throw std::runtime_error("Saved tensor data ended early for shape " + tensor.getShape().toString());
    }
}

} // namespace smart_dnn

#endif // TENSOR_IO_HPP

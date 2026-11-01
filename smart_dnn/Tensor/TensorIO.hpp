#ifndef TENSOR_IO_HPP
#define TENSOR_IO_HPP

#include "smart_dnn/Tensor/Tensor.hpp"
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace smart_dnn {

// Writes the tensor's rank, dimensions and raw values in native byte order.
template <typename T>
void saveTensor(std::ostream& os, const Tensor<T>& tensor) {
    const std::vector<int>& dims = tensor.getShape().getDimensions();
    const uint32_t rank = static_cast<uint32_t>(dims.size());
    os.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    os.write(reinterpret_cast<const char*>(dims.data()), static_cast<std::streamsize>(dims.size() * sizeof(int)));
    os.write(reinterpret_cast<const char*>(tensor.getData().data()), static_cast<std::streamsize>(tensor.getShape().size() * sizeof(T)));
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
    is.read(reinterpret_cast<char*>(tensor.getData().data()), static_cast<std::streamsize>(tensor.getShape().size() * sizeof(T)));
    if (!is) {
        throw std::runtime_error("Saved tensor data ended early for shape " + tensor.getShape().toString());
    }
}

} // namespace smart_dnn

#endif // TENSOR_IO_HPP

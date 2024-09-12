#ifndef TENSOR_FACTORY_CPU_IMPL_HPP
#define TENSOR_FACTORY_CPU_IMPL_HPP

#include "Tensor.hpp"
#include "TensorOperations.hpp"

namespace sdnn {

// Specialization for CPUDevice

// Generate a tensor filled with ones
template <typename T>
Tensor<T, CPUDevice> TensorFactory<T, CPUDevice>::ones(Shape dimensions) {
    return TensorOperations<T, CPUDevice>::createFill(dimensions, T(1));
}

template <typename T>
Tensor<T, CPUDevice> TensorFactory<T, CPUDevice>::zeros(Shape dimensions) {
    return TensorOperations<T, CPUDevice>::createFill(dimensions, T(0));
}

// Generate a tensor filled with random values in range [0, 1]
template <typename T>
Tensor<T, CPUDevice> TensorFactory<T, CPUDevice>::rand(Shape dimensions) {
    return TensorOperations<T, CPUDevice>::createRandom(dimensions, T(0), T(1));
}

// Generate a tensor filled with random values in range [min, max]
template <typename T>
Tensor<T, CPUDevice> TensorFactory<T, CPUDevice>::randn(Shape dimensions, T min, T max) {
    return TensorOperations<T, CPUDevice>::createRandom(dimensions, min, max);
}

template <typename T>
Tensor<T, CPUDevice> TensorFactory<T, CPUDevice>::identity(int size) {
    return TensorOperations<T, CPUDevice>::createIdentity(size);
}

} // namespace sdnn

#endif // TENSOR_FACTORY_CPU_IMPL_HPP

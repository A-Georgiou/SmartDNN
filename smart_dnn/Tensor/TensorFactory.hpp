#ifndef TENSOR_FACTORY_HPP
#define TENSOR_FACTORY_HPP

#include "smart_dnn/Tensor/Tensor.hpp"

namespace smart_dnn {

template <typename T, typename DeviceType>
class Tensor;  // Forward declaration of Tensor

namespace TensorFactory {

// Generate a tensor filled with ones
template <typename T, typename DeviceType>
Tensor<T, DeviceType> ones(Shape dimensions) {
    return TensorOperations<T, DeviceType>::createFill(dimensions, T(1));
}

template <typename T, typename DeviceType>
Tensor<T, DeviceType> ones(int size) {
    return ones<T, DeviceType>(Shape({size}));
}

// Generate a tensor filled with zeros
template <typename T, typename DeviceType>
Tensor<T, DeviceType> zeros(Shape dimensions) {
    return TensorOperations<T, DeviceType>::createFill(dimensions, T(0));
}

template <typename T, typename DeviceType>
Tensor<T, DeviceType> zeros(int size) {
    return zeros<T, DeviceType>(Shape({size}));
}

// Generate a tensor filled with random values in range [0, 1]
template <typename T, typename DeviceType>
Tensor<T, DeviceType> rand(Shape dimensions) {
    return TensorOperations<T, DeviceType>::createRandom(dimensions, T(0), T(1));
}

template <typename T, typename DeviceType>
Tensor<T, DeviceType> rand(int size) {
    return rand<T, DeviceType>(Shape({size}));
}

// Generate a tensor filled with random values in range [min, max]
template <typename T, typename DeviceType>
Tensor<T, DeviceType> randn(Shape dimensions, T min, T max) {
    return TensorOperations<T, DeviceType>::createRandom(dimensions, min, max);
}

template <typename T, typename DeviceType>
Tensor<T, DeviceType> randn(int size, T min, T max) {
    return randn<T, DeviceType>(Shape({size}), min, max);
}

template <typename T, typename DeviceType>
Tensor<T, DeviceType> identity(int size) {
    return TensorOperations<T, DeviceType>::createIdentity(size);
}

} // namespace TensorFactory

} // namespace smart_dnn

#endif // TENSOR_FACTORY_HPP
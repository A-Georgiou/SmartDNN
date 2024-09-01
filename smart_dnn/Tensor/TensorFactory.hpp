#ifndef TENSOR_FACTORY_HPP
#define TENSOR_FACTORY_HPP

#include "Tensor.hpp"

namespace smart_dnn {

template <typename T, typename DeviceType>
class Tensor;  // Forward declaration of Tensor

template <typename T, typename DeviceType>
class TensorFactory {
public:
    TensorFactory() = delete; // Prevent instantiation

    // Generate a tensor filled with ones
    static Tensor<T, DeviceType> ones(Shape dimensions) {
        return TensorOperations<T, DeviceType>::createFill(dimensions, T(1));
    }

    static Tensor<T, DeviceType> ones(size_t size){
        return ones(Shape({size}));
    }
    
    // Generate a tensor filled with zeros
    static Tensor<T, DeviceType> zeros(Shape dimensions) {
        return TensorOperations<T, DeviceType>::createFill(dimensions, T(0));
    }

    static Tensor<T, DeviceType> zeros(size_t size){
        return zeros(Shape({size}));
    }

    // Generate a tensor filled with random values in range [0, 1]
    static Tensor<T, DeviceType> rand(Shape dimensions) {
        return TensorOperations<T, DeviceType>::createRandom(dimensions, T(0), T(1));
    }

    static Tensor<T, DeviceType> rand(size_t size){
        return rand(Shape({size}));
    }

    // Generate a tensor filled with random values in range [min, max]
    static Tensor<T, DeviceType> randn(Shape dimensions, T min, T max) {
        return TensorOperations<T, DeviceType>::createRandom(dimensions, min, max);
    }

    static Tensor<T, DeviceType> randn(size_t size, T min, T max){
        return randn(Shape({size}), min, max);
    }

private:


};

}; // namespace smart_dnn

#endif // TENSOR_FACTORY_HPP
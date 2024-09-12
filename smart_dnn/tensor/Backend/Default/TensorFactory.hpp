#ifndef TENSOR_FACTORY_HPP
#define TENSOR_FACTORY_HPP

#include "Tensor.hpp"

namespace sdnn {

template <typename T, typename DeviceType>
class Tensor;  // Forward declaration of Tensor

template <typename T, typename DeviceType>
struct TensorFactory {
    static Tensor<T, DeviceType> ones(int size){
        return ones(Shape({size}));
    }
    static Tensor<T, DeviceType> zeros(int size){
        return zeros(Shape({size}));
    }
    static Tensor<T, DeviceType> rand(int size){
        return rand(Shape({size}));
    }
    static Tensor<T, DeviceType> randn(int size, T min, T max){
        return randn(Shape({size}), min, max);
    }
    static Tensor<T, DeviceType> identity(int size);
    static Tensor<T, DeviceType> ones(Shape dimensions);
    static Tensor<T, DeviceType> zeros(Shape dimensions);
    static Tensor<T, DeviceType> rand(Shape dimensions);
    static Tensor<T, DeviceType> randn(Shape dimensions, T min, T max);
};

template <typename T>
struct TensorFactory<T, CPUDevice> {
    static Tensor<T, CPUDevice> ones(Shape dimensions);
    static Tensor<T, CPUDevice> zeros(Shape dimensions);
    static Tensor<T, CPUDevice> rand(Shape dimensions);
    static Tensor<T, CPUDevice> randn(Shape dimensions, T min, T max);
    static Tensor<T, CPUDevice> identity(int size);
};

template <typename T>
struct TensorFactory<T, GPUDevice> {
    static Tensor<T, GPUDevice> ones(Shape dimensions);
    static Tensor<T, GPUDevice> zeros(Shape dimensions);
    static Tensor<T, GPUDevice> rand(Shape dimensions);
    static Tensor<T, GPUDevice> randn(Shape dimensions, T min, T max);
    static Tensor<T, GPUDevice> identity(int size);
};

} // namespace sdnn

#include "TensorFactoryCPU.impl.hpp"
#include "TensorFactoryGPU.impl.hpp"

#endif // TENSOR_FACTORY_HPP
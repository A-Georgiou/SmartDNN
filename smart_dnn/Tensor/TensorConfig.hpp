// TensorConfig.hpp
#ifndef TENSOR_CONFIG_HPP
#define TENSOR_CONFIG_HPP

#include "DeviceTypes.hpp"

namespace smart_dnn {

// Default configuration
template <typename T = float, typename DeviceType = CPUDevice>
struct TensorTraits {
    using DataType = T;
    using Device = DeviceType;
};

// User configuration struct, adjust to your needs
template <>
struct TensorTraits<> {
    using DataType = float;        // Default to float
    using Device = CPUDevice;      // Default to CPU
};

// Helper alias to easily get the current configuration
using CurrentTensorTraits = TensorTraits<>;

// Helper alias for the configured Tensor type
template <typename T = CurrentTensorTraits::DataType, 
          typename DeviceType = CurrentTensorTraits::Device>
using ConfiguredTensor = Tensor<T, DeviceType>;

} // namespace smart_dnn

#endif // TENSOR_CONFIG_HPP
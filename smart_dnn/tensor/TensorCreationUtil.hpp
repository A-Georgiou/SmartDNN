#ifndef TENSOR_CREATION_UTIL_HPP
#define TENSOR_CREATION_UTIL_HPP

#include <memory>
#include "smart_dnn/tensor/TensorAdapterBase.hpp"

#if USE_ARRAYFIRE_TENSORS
    #include "smart_dnn/tensor/Backend/ArrayFire/GPUTensor.hpp"
#else
    #include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"
#endif

namespace sdnn {

    // Variadic template function to create a tensor adapter
    template <typename... Args>
    inline std::unique_ptr<TensorAdapter> createTensorAdapter(Args&&... args) {
    #if USE_ARRAYFIRE_TENSORS
        std::cout << "Creating GPUTensor" << std::endl;
        return std::make_unique<GPUTensor>(std::forward<Args>(args)...);
    #else
        std::cout << "Creating CPUTensor" << std::endl;
        return std::make_unique<CPUTensor>(std::forward<Args>(args)...);
    #endif
    }

} // namespace sdnn

#endif // TENSOR_CREATION_UTIL_HPP
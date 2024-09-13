#ifndef TENSOR_CREATION_UTIL_HPP
#define TENSOR_CREATION_UTIL_HPP

#include <memory>
#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"

namespace sdnn {

    // Variadic template function to create a tensor adapter
    template <typename... Args>
    inline std::unique_ptr<TensorAdapter> createTensorAdapter(Args&&... args) {
    #if USE_ARRAYFIRE_TENSORS
        return std::make_unique<ArrayFireTensorAdapter>(std::forward<Args>(args)...);
    #else
        return std::make_unique<CPUTensor>(std::forward<Args>(args)...);
    #endif
    }

} // namespace sdnn

#endif // TENSOR_CREATION_UTIL_HPP
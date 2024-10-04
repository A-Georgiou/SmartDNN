#include "smart_dnn/tensor/TensorBackendUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensorBackend.hpp"
#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.hpp"

namespace sdnn {
    TensorBackend& defaultTensorBackend()
    {
        #if USE_ARRAYFIRE_TENSORS
            static GPUTensorBackend instance;
        #else
            static CPUTensorBackend instance;
        #endif
        return instance;
    }
}
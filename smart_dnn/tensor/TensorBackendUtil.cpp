#include "smart_dnn/tensor/TensorBackendUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensorBackend.hpp"
#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.hpp"
#include "smart_dnn/tensor/Backend/Eigen/EigenTensorBackend.hpp"

namespace sdnn {
    TensorBackend& defaultTensorBackend()
    {
        #if USE_ARRAYFIRE_TENSORS
            static GPUTensorBackend instance;
        #elif USE_EIGEN_TENSORS
            static EigenTensorBackend instance;
        #else
            static CPUTensorBackend instance;
        #endif
        return instance;
    }
}
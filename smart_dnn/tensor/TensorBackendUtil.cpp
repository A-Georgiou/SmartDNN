#include "smart_dnn/tensor/TensorBackendUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"

#if USE_ARRAYFIRE_TENSORS
    #include "smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.hpp"
    #include <arrayfire.h>
#elif USE_EIGEN_TENSORS
    #include "smart_dnn/tensor/Backend/Eigen/EigenTensorBackend.hpp"
#else
    #include "smart_dnn/tensor/Backend/Default/CPUTensorBackend.hpp"
#endif

namespace sdnn {
    TensorBackend& defaultTensorBackend()
    {
        #if USE_ARRAYFIRE_TENSORS
            // Initialize ArrayFire with available backend
            static bool arrayfire_initialized = false;
            if (!arrayfire_initialized) {
                try {
                    // Try to get available backends and use CPU if available
                    int backends = af::getAvailableBackends();
                    if (backends & AF_BACKEND_CPU) {
                        af::setBackend(AF_BACKEND_CPU);
                    } else if (backends & AF_BACKEND_OPENCL) {
                        af::setBackend(AF_BACKEND_OPENCL);
                    } else if (backends & AF_BACKEND_CUDA) {
                        af::setBackend(AF_BACKEND_CUDA);
                    } else {
                        throw std::runtime_error("No ArrayFire backends available");
                    }
                    arrayfire_initialized = true;
                } catch (const af::exception& e) {
                    throw std::runtime_error("Failed to initialize ArrayFire backend: " + std::string(e.what()));
                }
            }
            static GPUTensorBackend instance;
        #elif USE_EIGEN_TENSORS
            static EigenTensorBackend instance;
        #else
            static CPUTensorBackend instance;
        #endif
        return instance;
    }
}
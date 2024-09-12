#include "smart_dnn/tensor/TensorBackendUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensorBackend.hpp"

namespace sdnn {

    TensorBackend& defaultTensorBackend()
    {
        static CPUTensorBackend instance;
        return instance;
    }
}
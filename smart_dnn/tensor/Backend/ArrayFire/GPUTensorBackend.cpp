#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.hpp"

namespace sdnn {

    Tensor GPUTensorBackend::fill(const Shape& shape, const DataItem& value, dtype type) const {
        return Tensor(shape, value, type);
    }

}
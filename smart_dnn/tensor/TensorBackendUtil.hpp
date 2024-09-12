// TensorBackendUtils.hpp
#ifndef TENSOR_BACKEND_UTIL_HPP
#define TENSOR_BACKEND_UTIL_HPP

#include "smart_dnn/tensor/TensorBackend.hpp"

namespace sdnn {

// Function to return the default tensor backend
TensorBackend& defaultTensorBackend();

} // namespace sdnn

#endif // TENSOR_BACKEND_UTIL_HPP
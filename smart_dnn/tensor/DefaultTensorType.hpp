#ifndef DEFAULT_TENSOR_TYPE_HPP
#define DEFAULT_TENSOR_TYPE_HPP

#if USE_ARRAYFIRE_TENSORS
  #include "Backend/ArrayFire/GPUTensor.hpp"
  #include "Backend/ArrayFire/GPUBackend.hpp"
#endif

#endif // DEFAULT_TENSOR_TYPE_HPP

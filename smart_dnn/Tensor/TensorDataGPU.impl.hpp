#ifndef TENSOR_DATA_GPU_IMPL_HPP
#define TENSOR_DATA_GPU_IMPL_HPP

namespace smart_dnn {

// Specialization for GPUDevice
#define TEMPLATE_TENSOR template <typename T>
#define TENSOR_DATA_GPU TensorData<T, GPUDevice>

/*

TODO: Implement the respective TensorData methods here.

*/

// Clean up macro definitions
#undef TEMPLATE_TENSOR
#undef TENSOR_DATA_GPU

}; // namespace smart_dnn

#endif // TENSOR_DATA_GPU_IMPL_HPP

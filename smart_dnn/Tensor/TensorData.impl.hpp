#ifndef TENSOR_DATA_IMPL_HPP
#define TENSOR_DATA_IMPL_HPP

namespace smart_dnn {

// Specialization for CPUDevice
#define TEMPLATE_TENSOR template <typename T>
#define TENSOR_DATA_CPU TensorData<T, CPUDevice>

TEMPLATE_TENSOR
TENSOR_DATA_CPU::TensorData(Shape dimensions) noexcept 
    : shape_(std::move(dimensions)), data_(std::make_unique<T[]>(shape_.size())) {}

TEMPLATE_TENSOR
TENSOR_DATA_CPU::TensorData(Shape dimensions, T value) noexcept 
    : shape_(dimensions), data_(std::make_unique<T[]>(dimensions.size())) {
    std::fill_n(data_.get(), shape_.size(), value);
}

TEMPLATE_TENSOR
TENSOR_DATA_CPU::TensorData(Shape dimensions, const T* data) 
    : shape_(dimensions), data_(std::make_unique<T[]>(dimensions.size())) {
    std::copy_n(data, shape_.size(), data_.get());
}

TEMPLATE_TENSOR
TENSOR_DATA_CPU::TensorData(const TENSOR_DATA_CPU& other) 
    : shape_(other.shape_), data_(std::make_unique<T[]>(other.shape_.size())) {
    std::copy_n(other.data_.get(), shape_.size(), data_.get());
}

TEMPLATE_TENSOR
TENSOR_DATA_CPU& TENSOR_DATA_CPU::operator=(const TENSOR_DATA_CPU&) {
    if (this != &other) {
        shape_ = other.shape_;
        data_.reset(new T[shape_.size()]);
        std::copy_n(other.data_.get(), shape_.size(), data_.get());
    }
    return *this;
}

// Clean up macro definitions
#undef TEMPLATE_TENSOR_DATA
#undef TEMPLATE_TENSOR
#undef TENSOR_DATA
#undef TENSOR_DATA_CPU

}; // namespace smart_dnn

#endif // TENSOR_DATA_IMPL_HPP

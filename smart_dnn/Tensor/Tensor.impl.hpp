#ifndef TENSOR_IMPL_HPP
#define TENSOR_IMPL_HPP

#include "DeviceTypes.hpp"

namespace smart_dnn {

#define TEMPLATE_TENSOR template <typename T, typename DeviceType>

TEMPLATE_TENSOR
Tensor<T, DeviceType>::Tensor(Shape dimensions) noexcept : data_(dimensions) {}

TEMPLATE_TENSOR
Tensor<T, DeviceType>::Tensor(Shape dimensions, T value) noexcept : data_(dimensions, value) {}

TEMPLATE_TENSOR
Tensor<T, DeviceType>::Tensor(Shape dimensions, const T* dataArray) : data_(dimensions, dataArray) {}

TEMPLATE_TENSOR
Tensor<T, DeviceType>::Tensor(const Tensor<T, DeviceType>& other) : data_(other.data_) {}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator=(Tensor<T, DeviceType>&& other) noexcept {
    data_ = std::move(other.data_);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator=(const Tensor<T, DeviceType>& other) {
    if (this != &other) {
        data_ = other.data_;
    }
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator+=(const Tensor<T, DeviceType>& other) {
    TensorOperations<T, DeviceType>::add(data_, other.data_);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator-=(const Tensor<T, DeviceType>& other) {
    TensorOperations<T, DeviceType>::subtract(data_, other.data_);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator*=(const Tensor<T, DeviceType>& other) {
    TensorOperations<T, DeviceType>::multiply(data_, other.data_);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator/=(const Tensor<T, DeviceType>& other) {
    TensorOperations<T, DeviceType>::divide(data_, other.data_);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator+=(T scalar) {
    TensorOperations<T, DeviceType>::addScalar(data_, scalar);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator-=(T scalar) {
    TensorOperations<T, DeviceType>::subtractScalar(data_, scalar);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator*=(T scalar) {
    TensorOperations<T, DeviceType>::multiplyScalar(data_, scalar);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator/=(T scalar) {
    TensorOperations<T, DeviceType>::divideScalar(data_, scalar);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator+(const Tensor<T, DeviceType>& other) const {
    Tensor<T, DeviceType> result(data_.shape());
    TensorOperations<T, DeviceType>::add(result.data_, other.data_);
    return result;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator-(const Tensor<T, DeviceType>& other) const {
    Tensor<T, DeviceType> result(data_.shape());
    TensorOperations<T, DeviceType>::subtract(result.data_, other.data_);
    return result;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator*(const Tensor<T, DeviceType>& other) const {
    Tensor<T, DeviceType> result(data_.shape());
    TensorOperations<T, DeviceType>::multiply(result.data_, other.data_);
    return result;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator/(const Tensor<T, DeviceType>& other) const {
    Tensor<T, DeviceType> result(data_.shape());
    TensorOperations<T, DeviceType>::divide(result.data_, other.data_);
    return result;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator+(T scalar) const {
    Tensor<T, DeviceType> result(data_.shape());
    TensorOperations<T, DeviceType>::addScalar(result.data_, scalar);
    return result;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator-(T scalar) const {
    Tensor<T, DeviceType> result(data_.shape());
    TensorOperations<T, DeviceType>::subtractScalar(result.data_, scalar);
    return result;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator*(T scalar) const {
    Tensor<T, DeviceType> result(data_.shape());
    TensorOperations<T, DeviceType>::multiplyScalar(result.data_, scalar);
    return result;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator/(T scalar) const {
    Tensor<T, DeviceType> result(data_.shape());
    TensorOperations<T, DeviceType>::divideScalar(result.data_, scalar);
    return result;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator-() const {
    Tensor<T, DeviceType> result(data_.shape());
    TensorOperations<T, DeviceType>::subtractScalar(result.data_, 0);
    return result;
}

TEMPLATE_TENSOR
TensorData<T, DeviceType> Tensor<T, DeviceType>::getData() const noexcept {
    return data_;
}

TEMPLATE_TENSOR
Shape Tensor<T, DeviceType>::getShape() const noexcept {
    return data_.shape();
}

TEMPLATE_TENSOR
bool Tensor<T, DeviceType>::operator==(const Tensor<T, DeviceType>& other) const {
    return data_ == other.data_;
}

TEMPLATE_TENSOR
bool Tensor<T, DeviceType>::operator!=(const Tensor<T, DeviceType>& other) const {
    return data_ != other.data_;
}

// Non-member operator overloads
TEMPLATE_TENSOR
Tensor<T, DeviceType> operator+(T scalar, const Tensor<T, DeviceType>& tensor) {
    return tensor + scalar;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> operator-(T scalar, const Tensor<T, DeviceType>& tensor) {
    return -tensor + scalar;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> operator*(T scalar, const Tensor<T, DeviceType>& tensor) {
    return tensor * scalar;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> operator/(T scalar, const Tensor<T, DeviceType>& tensor) {
    Tensor<T, DeviceType> result(tensor.getShape());
    return result;
}

#undef TEMPLATE_TENSOR

} // namespace smart_dnn

#endif // TENSOR_IMPL_HPP

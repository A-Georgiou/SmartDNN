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
Tensor<T, DeviceType>::Tensor(const TensorData<T, DeviceType>& data) noexcept : data_(data) {}

TEMPLATE_TENSOR
Tensor<T, DeviceType>::Tensor(TensorData<T, DeviceType>&& data) noexcept : data_(std::move(data)) {}

TEMPLATE_TENSOR
Tensor<T, DeviceType>::Tensor(const Tensor<T, DeviceType>& other) : data_(other.data_) {}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator=(const Tensor<T, DeviceType>& other) {
    if (this != &other) {
        data_ = other.data_;
    }
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator+=(const Tensor<T, DeviceType>& other) {
    TensorOperations<T, DeviceType>::addInPlace(data_, other.data_);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator-=(const Tensor<T, DeviceType>& other) {
    TensorOperations<T, DeviceType>::subtractInPlace(data_, other.data_);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator*=(const Tensor<T, DeviceType>& other) {
    TensorOperations<T, DeviceType>::multipleInPlace(data_, other.data_);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator/=(const Tensor<T, DeviceType>& other) {
    TensorOperations<T, DeviceType>::divideInPlace(data_, other.data_);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator+=(T scalar) {
    TensorOperations<T, DeviceType>::addScalarInPlace(data_, scalar);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator-=(T scalar) {
    TensorOperations<T, DeviceType>::substractScalarInPlace(data_, scalar);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator*=(T scalar) {
    TensorOperations<T, DeviceType>::multipleScalarInPlace(data_, scalar);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::operator/=(T scalar) {
    TensorOperations<T, DeviceType>::divideScalarInPlace(data_, scalar);
    return *this;
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator+(const Tensor<T, DeviceType>& other) const {
    return TensorOperations<T, DeviceType>::add(this->data_, other.data_);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator-(const Tensor<T, DeviceType>& other) const {
    return TensorOperations<T, DeviceType>::subtract(this->data_, other.data_);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator*(const Tensor<T, DeviceType>& other) const {
    return TensorOperations<T, DeviceType>::multiply(this->data_, other.data_);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator/(const Tensor<T, DeviceType>& other) const {
    return TensorOperations<T, DeviceType>::divide(this->data_, other.data_);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator+(T scalar) const {
    return TensorOperations<T, DeviceType>::addScalar(this->data_, scalar);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator-(T scalar) const {
    return TensorOperations<T, DeviceType>::subtractScalar(this->data_, scalar);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator*(T scalar) const {
    return TensorOperations<T, DeviceType>::multiplyScalar(this->data_, scalar);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator/(T scalar) const {
    return TensorOperations<T, DeviceType>::divideScalar(this->data_, scalar);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator-() const {
    return TensorOperations<T, DeviceType>::multiplyScalar(this->data_, -1);
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
    return TensorOperations<T, DeviceType>::inverseDivideScalar(tensor.getData(), scalar);
}

TEMPLATE_TENSOR
std::string Tensor<T, DeviceType>::detailedString() const {
    std::ostringstream oss;
    oss << "Tensor:\n";
    oss << data_.toString() << "\n";
    return oss.str();
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::ones(Shape dimensions) {
    return TensorFactory<T, DeviceType>::ones(dimensions);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::zeros(Shape dimensions) {
    return TensorFactory<T, DeviceType>::zeros(dimensions);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::rand(Shape dimensions) {
    return TensorFactory<T, DeviceType>::rand(dimensions);
}

#undef TEMPLATE_TENSOR

} // namespace smart_dnn

#endif // TENSOR_IMPL_HPP

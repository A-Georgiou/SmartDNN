#ifndef TENSOR_IMPL_HPP
#define TENSOR_IMPL_HPP

#include "DeviceTypes.hpp"

namespace sdnn {

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
Tensor<T, DeviceType>::Tensor(Tensor<T, DeviceType>&& other) noexcept : data_(std::move(other.data_)) {
    other.data_ = TensorData<T, DeviceType>(Shape({1})); // We should not have an empty tensor
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>::Tensor(Shape dimensions, std::initializer_list<T> values) : data_(dimensions, values) {}

TEMPLATE_TENSOR
Tensor<T, DeviceType>::Tensor(Shape dimensions, const std::vector<T>& values) : data_(dimensions, values) {}

TEMPLATE_TENSOR
Tensor<T, DeviceType>::Tensor(Shape dimensions, std::vector<T>&& values) noexcept : data_(dimensions, std::move(values)) {}

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
    TensorOperations<T, DeviceType>::multiplyInPlace(data_, other.data_);
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
T& Tensor<T, DeviceType>::operator[](size_t index){
    return data_[index];
}

TEMPLATE_TENSOR
const T& Tensor<T, DeviceType>::operator[](size_t index) const {
    return data_[index];
}
    
TEMPLATE_TENSOR
T& Tensor<T, DeviceType>::at(std::vector<int> indices) {
    return data_.at(indices);
}

TEMPLATE_TENSOR
const T& Tensor<T, DeviceType>::at(std::vector<int> indices) const {
    return data_.at(indices);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::operator-() const {
    return TensorOperations<T, DeviceType>::multiplyScalar(this->data_, -1);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::sqrt() const {
    return TensorOperations<T, DeviceType>::sqrt(this->data_);
}

TEMPLATE_TENSOR
TensorData<T, DeviceType>& Tensor<T, DeviceType>::getData() noexcept {
    return data_;
}

TEMPLATE_TENSOR
const TensorData<T, DeviceType>& Tensor<T, DeviceType>::getData() const noexcept {
    return data_;
}

TEMPLATE_TENSOR
const Shape& Tensor<T, DeviceType>::getShape() const noexcept {
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
std::string Tensor<T, DeviceType>::toDetailedString() const {
    std::ostringstream oss;
    oss << "Tensor:\n";
    oss << data_.toString() << "\n";
    return oss.str();
}

TEMPLATE_TENSOR
std::string Tensor<T, DeviceType>::toDataString() const {
    return data_.toDataString();
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

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::randn(Shape dimensions, T min, T max){
    return TensorFactory<T, DeviceType>::randn(dimensions, min, max);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::ones(int size) {
    return TensorFactory<T, DeviceType>::ones(size);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::zeros(int size) {
    return TensorFactory<T, DeviceType>::zeros(size);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::rand(int size) {
    return TensorFactory<T, DeviceType>::rand(size);
}

TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::randn(int size, T min, T max){
    return TensorFactory<T, DeviceType>::randn(size, min, max);
}


TEMPLATE_TENSOR
Tensor<T, DeviceType> Tensor<T, DeviceType>::identity(int size) {
    return TensorFactory<T, DeviceType>::identity(size);
}

TEMPLATE_TENSOR
void Tensor<T, DeviceType>::reshape(const Shape& newShape) {
    this->data_.reshape(newShape);
}

TEMPLATE_TENSOR
void Tensor<T, DeviceType>::reshape(const std::vector<int>& dims) {
    this->data_.reshape(dims);
}

TEMPLATE_TENSOR
void Tensor<T, DeviceType>::reshape(const std::initializer_list<int>& dims) {
    std::vector<int> dimsVec(dims);
    this->data_.reshape(dimsVec);
}

template <typename T, typename DeviceType>
Tensor<T, DeviceType> Tensor<T, DeviceType>::slice(int dim, int index) const {
    Shape newShape = this->data_.shape();
    std::vector<int> shape = newShape.getDimensions();
    const std::vector<size_t> strides = this->data_.stride();

    int offset = index * strides[dim];

    shape.erase(shape.begin() + dim);

    TensorData<T, DeviceType> slicedData(Shape(shape), this->data_.data() + offset);

    return Tensor<T, DeviceType>(std::move(slicedData));
}

TEMPLATE_TENSOR
Tensor<T, DeviceType>& Tensor<T, DeviceType>::apply(std::function<T(T)> func) {
    TensorOperations<T, DeviceType>::applyInPlace(this->data_, func);
    return *this;  // Return the current Tensor object
}


#undef TEMPLATE_TENSOR

} // namespace sdnn

#endif // TENSOR_IMPL_HPP

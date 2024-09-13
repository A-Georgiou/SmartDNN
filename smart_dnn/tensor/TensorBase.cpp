#include "smart_dnn/tensor/TensorBase.hpp" 
#include "smart_dnn/tensor/TensorAdapterBase.hpp"  // Full definition required for std::unique_ptr
#include "smart_dnn/tensor/TensorBackend.hpp"
#include "smart_dnn/tensor/TensorBackendUtil.hpp"
#include "smart_dnn/tensor/TensorView.hpp"

namespace sdnn {
/*

HELPER FUNCTIONS

*/

template <typename T>
dtype dtypeFromType() {
    return dtype_trait<T>::value;
}

template <typename T>
bool Tensor::isSameType() const {
    return type() == dtype_trait<T>::value;
}

/*

TENSOR CLASS IMPLEMENTATION

*/

Tensor::~Tensor() = default;

Tensor::Tensor(std::unique_ptr<TensorAdapter> tensorImpl): tensorImpl_(std::move(tensorImpl)) {}

template <typename T>
Tensor::Tensor(const Shape& shape, const std::vector<T>& data)
    : tensorImpl_(createTensorAdapter(shape, data.data(), dtype_trait<T>::value)) {}

Tensor::Tensor(const Tensor& tensor)
    : tensorImpl_(tensor.tensorImpl_->clone()) {}

Tensor::Tensor(Tensor&& tensor) noexcept
    : tensorImpl_(std::move(tensor.tensorImpl_)) {}

Tensor& Tensor::operator=(const Tensor& tensor) {
    if (this != &tensor) {
        tensorImpl_ = tensor.tensorImpl_->clone();
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& tensor) noexcept {
    if (this != &tensor) {
        tensorImpl_ = std::move(tensor.tensorImpl_);
    }
    return *this;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    tensorImpl_->addInPlace(other);
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    tensorImpl_->subtractInPlace(other);
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    tensorImpl_->multiplyInPlace(other);
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    tensorImpl_->divideInPlace(other);
    return *this;
}

Tensor& Tensor::operator+=(const double& scalar) {
    tensorImpl_->addScalarInPlace(scalar);
    return *this;
}

Tensor& Tensor::operator-=(const double& scalar) {
    tensorImpl_->subtractScalarInPlace(scalar);
    return *this;
}

Tensor& Tensor::operator*=(const double& scalar) {
    tensorImpl_->multiplyScalarInPlace(scalar);
    return *this;
}  

Tensor& Tensor::operator/=(const double& scalar) {
    tensorImpl_->divideScalarInPlace(scalar);
    return *this;
}

TensorView Tensor::operator[](const std::initializer_list<size_t>& indices) {
    return tensorImpl_->at(indices);
}

TensorView Tensor::operator[](const std::vector<size_t>& indices) {
    return tensorImpl_->at(indices);
}


bool Tensor::operator==(const Tensor& other) const{
    return tensorImpl_->equal(other);
}

bool Tensor::operator!=(const Tensor& other) const{
    return !(tensorImpl_->equal(other));
}

template <typename T>
T Tensor::at(size_t index) const {
    if (!isSameType<T>()) {
        throw std::invalid_argument(
            "Tensor::at: requested type does not match tensor type"
        );
    }
    T out;
    tensorImpl_->at(index, &out);
    return out;
}

template <typename T>
T Tensor::at(const std::vector<size_t>& indices) const {
    if (!isSameType<T>()) {
        throw std::invalid_argument(
            "Tensor::at: requested type does not match tensor type");
    }
    T out;
    tensorImpl_->at(indices, &out);
    return out;
}

void Tensor::set(size_t index, const double& value) {
    tensorImpl_->set(index, value);
}

void Tensor::set(const std::vector<size_t>& indices, const double& value) {
    tensorImpl_->set(indices, value);
}

const Shape& Tensor::shape() const noexcept {
    return tensorImpl_->shape();
}

dtype Tensor::type() const noexcept {
    return tensorImpl_->type();
}

const TensorBackend& Tensor::backend() const {
    return tensorImpl_->backend();
}

std::string Tensor::toString() const {
    return tensorImpl_->toString();
}

std::string Tensor::toDataString() const {
    return tensorImpl_->toDataString();
}


/*

FREE FUNCTION IMPLEMENTATIONS

*/

Tensor add(const Tensor& lhs, const Tensor& rhs) {
    return lhs.backend().add(lhs, rhs);
}

Tensor sub(const Tensor& lhs, const Tensor& rhs) {
    return lhs.backend().sub(lhs, rhs);
}

Tensor mul(const Tensor& lhs, const Tensor& rhs) {
    return lhs.backend().mul(lhs, rhs);
}

Tensor div(const Tensor& lhs, const Tensor& rhs) {
    return lhs.backend().div(lhs, rhs);
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    return add(lhs, rhs);
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    return sub(lhs, rhs);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    return mul(lhs, rhs);
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    return div(lhs, rhs);
}

Tensor add(const Tensor& tensor, const double& scalar) {
    return tensor.backend().add(tensor, scalar);
}

Tensor sub(const Tensor& tensor, const double& scalar) {
    return tensor.backend().sub(tensor, scalar);
}

Tensor mul(const Tensor& tensor, const double& scalar) {
    return tensor.backend().mul(tensor, scalar);
}

Tensor div(const Tensor& tensor, const double& scalar) {
    return tensor.backend().div(tensor, scalar);
}

Tensor operator+(const Tensor& tensor, const double& scalar) {
    return add(tensor, scalar);
}

Tensor operator-(const Tensor& tensor, const double& scalar) {
    return sub(tensor, scalar);
}

Tensor operator*(const Tensor& tensor, const double& scalar) {
    return mul(tensor, scalar);
}

Tensor operator/(const Tensor& tensor, const double& scalar) {
    return div(tensor, scalar);
}

Tensor operator+(const double& scalar, const Tensor& tensor) {
    return add(tensor, scalar);
}

Tensor operator-(const double& scalar, const Tensor& tensor) {
    return tensor.backend().scalarSub(scalar, tensor);
}

Tensor operator*(const double& scalar, const Tensor& tensor) {
    return mul(tensor, scalar);
}

Tensor operator/(const double& scalar, const Tensor& tensor) {
    return tensor.backend().scalarDiv(scalar, tensor);
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    return lhs.backend().matmul(lhs, rhs);
}

Tensor reshape(const Tensor& tensor, const Shape& newShape) {
    return tensor.backend().reshape(tensor, newShape);
}

Tensor transpose(const Tensor& tensor, const std::vector<int>& axes) {
    return tensor.backend().transpose(tensor, axes);
}

Tensor sum(const Tensor& input, const std::vector<int>& axes, bool keepDims){
    return input.backend().sum(input, axes, keepDims);
}

Tensor mean(const Tensor& input, const std::vector<int>& axes, bool keepDims){
    return input.backend().mean(input, axes, keepDims);
}

Tensor zeros(const Shape& shape, dtype type) {
    return defaultTensorBackend().fill(shape, 0.0, type);
}

Tensor ones(const Shape& shape, dtype type) {
    return defaultTensorBackend().fill(shape, 1.0, type);
}

Tensor rand(const Shape& shape, dtype type) {
    return defaultTensorBackend().rand(shape, type);
}

Tensor fill(const Shape& shape, dtype type, const double& fillValue) {
    std::cout << "fill" << std::endl;
    return defaultTensorBackend().fill(shape, fillValue, type);
}

}; // namespace sdnn


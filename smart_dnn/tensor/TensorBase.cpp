#include "smart_dnn/tensor/TensorBase.hpp" 
#include "smart_dnn/tensor/TensorAdapterBase.hpp"  // Full definition required for std::unique_ptr
#include "smart_dnn/tensor/TensorBackend.hpp"

namespace sdnn {
    
/*

TENSOR CLASS IMPLEMENTATION

*/

Tensor::~Tensor() = default;

Tensor::Tensor(std::unique_ptr<TensorAdapter> tensorImpl): tensorImpl_(std::move(tensorImpl)) {}

Tensor::Tensor(const Tensor& tensor)
    : tensorImpl_(tensor.tensorImpl_->clone()) {}

Tensor::Tensor(Tensor&& tensor) noexcept
    : tensorImpl_(std::move(tensor.tensorImpl_)) {
        tensor.tensorImpl_ = nullptr;
    }

Tensor& Tensor::operator=(const Tensor& tensor) {
    if (this != &tensor) {
        tensorImpl_ = tensor.tensorImpl_->clone();
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& tensor) noexcept {
    if (this != &tensor) {
        tensorImpl_ = std::move(tensor.tensorImpl_);
        tensor.tensorImpl_ = nullptr;
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

Tensor Tensor::operator[](const std::initializer_list<size_t>& indices) {
    return tensorImpl_->at(indices);
}

Tensor Tensor::operator[](const std::vector<size_t>& indices) {
    return tensorImpl_->at(indices);
}

Tensor Tensor::operator[](size_t index) {
    return tensorImpl_->at(index);
}

const Tensor Tensor::operator[](size_t index) const {
    return tensorImpl_->at(index);
}

bool Tensor::operator==(const Tensor& other) const{
    return tensorImpl_->equal(other);
}

bool Tensor::operator!=(const Tensor& other) const{
    return !(tensorImpl_->equal(other));
}

Tensor Tensor::clone() const {
    return Tensor(tensorImpl_->clone());
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

void Tensor::apply(const std::function<void(double&)>& func) {
    tensorImpl_->apply(func);
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

Tensor apply(const Tensor& tensor, const std::function<void(double&)>& func) {
    return tensor.backend().apply(tensor, func);
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

Tensor sqrt(const Tensor& tensor) {
    return tensor.backend().sqrt(tensor);
}

Tensor sum(const Tensor& input, const std::vector<int>& axes, bool keepDims){
    return input.backend().sum(input, axes, keepDims);
}

Tensor mean(const Tensor& input, const std::vector<int>& axes, bool keepDims){
    return input.backend().mean(input, axes, keepDims);
}

Tensor zeros(const Shape& shape, dtype type) {
    int16_t zero = 0;
    return defaultTensorBackend().fill(shape, {&zero, dtype::s16}, type);
}

Tensor ones(const Shape& shape, dtype type) {
    int16_t one = 1;
    return defaultTensorBackend().fill(shape, {&one, dtype::s16}, type);
}

Tensor rand(const Shape& shape, dtype type) {
    return defaultTensorBackend().rand(shape, type);
}

}; // namespace sdnn


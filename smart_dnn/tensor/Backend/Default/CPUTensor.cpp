

#include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"
#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensorBackend.hpp"
#include "smart_dnn/shape/Shape.hpp"
#include "smart_dnn/tensor/TensorView.hpp"
#include <typeindex>
#include <memory>

namespace sdnn {

CPUTensor::~CPUTensor() {
    data_.clear();
}

CPUTensor::CPUTensor(const Shape& shape, dtype type)
    : shape_(shape), type_(type) {
    allocateMemory(shape.size() * dtype_size(type));
}

CPUTensor::CPUTensor(const Shape& shape, const double* data, dtype type)
    : shape_(shape), type_(type) {
    size_t total_elements = shape.size();
    allocateMemory(total_elements * dtype_size(type));

    for (size_t i = 0; i < total_elements; ++i) {
        applyTypedOperationHelper(type_, [this, &data, i](auto dummy) {
            using TargetType = decltype(dummy);
            writeElement<TargetType>(data_.data(), i, data[i]);
        });
    }
}

CPUTensor::CPUTensor(const Shape& shape, const std::vector<double>& data, dtype type)
    : shape_(shape), type_(type) {
    if (shape.size() != data.size()) {
        throw std::invalid_argument("Data size does not match shape");
    }
    allocateMemory(shape.size() * dtype_size(type));
    for (size_t i = 0; i < shape.size(); ++i){
        applyTypedOperationHelper(type, [this, &data, i](auto dummy) {
            using TargetType = decltype(dummy);
            writeElement<TargetType>(data_.data(), i, data[i]);
        });
    }
}

CPUTensor::CPUTensor(const CPUTensor& other){
    shape_ = other.shape_;
    type_ = other.type_;
    allocateMemory(shape_.size() * dtype_size(type_));
    std::memcpy(data_.data(), other.data_.data(), shape_.size() * dtype_size(type_));
}

CPUTensor::CPUTensor(CPUTensor&& other) noexcept {
    shape_ = std::move(other.shape_);
    type_ = std::move(other.type_);
    data_ = std::move(other.data_);
}


CPUTensor& CPUTensor::operator=(const CPUTensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        type_ = other.type_;
        allocateMemory(shape_.size() * dtype_size(type_));
        std::memcpy(data_.data(), other.data_.data(), shape_.size() * dtype_size(type_));
    }
    return *this;
}

CPUTensor& CPUTensor::operator=(CPUTensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        type_ = std::move(other.type_);
        data_ = std::move(other.data_);
    }
    return *this;
}

TensorView CPUTensor::operator[](size_t index) {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return TensorView(*this, index);
}

const TensorView CPUTensor::operator[](size_t index) const {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return TensorView(const_cast<CPUTensor&>(*this), index);
}

void CPUTensor::set(const std::vector<size_t>& indices, const double& value) {
    size_t flatIndex = computeFlatIndex(shape_, indices);
    setValueFromDouble(flatIndex, value);
}

void CPUTensor::set(size_t index, const double& value){
    setValueFromDouble(index, value);
}

TensorView CPUTensor::at(const std::vector<size_t>& indices) const {
    size_t flatIndex = computeFlatIndex(shape_, indices);
    return TensorView(const_cast<CPUTensor&>(*this), flatIndex);
 }

void CPUTensor::addInPlace(const Tensor& other) {
    elementWiseOperation(other, [](auto& a, const auto& b) { a += b; });
}

void CPUTensor::subtractInPlace(const Tensor& other) {
    elementWiseOperation(other, [](auto& a, const auto& b) { a -= b; });
}

void CPUTensor::multiplyInPlace(const Tensor& other) {
    elementWiseOperation(other, [](auto& a, const auto& b) { a *= b; });
}

void CPUTensor::divideInPlace(const Tensor& other) {
    elementWiseOperation(other, [](auto& a, const auto& b) { 
        if (b == 0) throw std::runtime_error("Division by zero");
        a /= b; 
    });
}

void CPUTensor::addScalarInPlace(double scalar) {
    scalarOperation(scalar, [](auto& a, double b) { a += b; });
}

void CPUTensor::subtractScalarInPlace(double scalar) {
    scalarOperation(scalar, [](auto& a, double b) { a -= b; });
}

void CPUTensor::multiplyScalarInPlace(double scalar) {
    scalarOperation(scalar, [](auto& a, double b) { a *= b; });
}

void CPUTensor::divideScalarInPlace(double scalar) {
    if (scalar == 0) throw std::runtime_error("Division by zero");
    scalarOperation(scalar, [](auto& a, double b) { a /= b; });
}

bool CPUTensor::equal(const Tensor& other) const {
    if (shape_ != other.shape() || type_ != other.type()) {
        return false;
    }
    return elementWiseComparison(other, [](auto a, auto b) { return a == b; });
}

bool CPUTensor::greaterThan(const Tensor& other) const {
    return elementWiseComparison(other, [](auto a, auto b) { return a > b; });
}

bool CPUTensor::lessThan(const Tensor& other) const {
    return elementWiseComparison(other, [](auto a, auto b) { return a < b; });
}

void CPUTensor::reshape(const Shape& newShape) {
    if (shape_.size() != newShape.size()) {
        throw std::runtime_error("Number of elements must remain constant during reshape");
    }
    shape_ = newShape;
}

std::unique_ptr<TensorAdapter> CPUTensor::clone() const {
    return std::make_unique<CPUTensor>(*this);
}

std::string CPUTensor::toString() {
    std::string result = "CPUTensor(" + shape_.toString() + ", " + dtypeToString(type_) + ")\n";
    result += toDataString();
    return result;
}

std::string CPUTensor::toDataString() {
    std::stringstream ss;
    applyTypedOperation([&ss, this](auto* dummy) {
        using T = std::remove_pointer_t<decltype(dummy)>;
        const auto* data = this->typedData<T>();
        for (size_t i = 0; i < this->shape_.size(); ++i) {
            if (i > 0) ss << " ";
            ss << data[i];
        }
    });
    return ss.str();
}

void CPUTensor::fill(const double& value) {
    scalarOperation(value, [](auto& a, double b) { a = b; });
}

inline void CPUTensor::allocateMemory(size_t size) {
    data_.resize(size);
}

TensorBackend& CPUTensor::backend() const {
    static CPUTensorBackend backend;
    return backend;
}

double CPUTensor::getValueAsDouble(size_t index) const {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range");
    }

    double result = 0.0;

    applyTypedOperationHelper(type_, [this, index, &result](auto dummy) {
        using T = decltype(dummy);
        const T* data = reinterpret_cast<const T*>(data_.data());
        result = static_cast<double>(data[index]);
    });

    return result;
}

void CPUTensor::setValueFromDouble(size_t index, double value) {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range");
    }

    applyTypedOperationHelper(type_, [this, index, value](auto dummy) {
        using T = decltype(dummy);
        T* data = reinterpret_cast<T*>(data_.data());
        data[index] = static_cast<T>(value);
    });
}


} // namespace sdnn


#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensor.hpp"
#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.hpp"
#include "smart_dnn/tensor/Backend/ArrayFire/Utils.hpp"
#include "smart_dnn/shape/Shape.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include <typeindex>
#include <memory>
#include <any>

namespace sdnn {

GPUTensor::GPUTensor(const Shape& shape, dtype type)
    : shape_(shape), type_(type) {
    std::vector<dim_t> af_dims(shape.getDimensions().begin(), shape.getDimensions().end());
    data_ = std::make_shared<af::array>(af_dims.size(), af_dims.data(), utils::sdnnToAfType(type));
}

GPUTensor::GPUTensor(const Shape& shape, const af::array& data, dtype type)
    : shape_(shape), type_(type) {
    if (shape.size() != data.elements()) {
        throw std::invalid_argument("Data size does not match shape");
    }
    data_ = std::make_shared<af::array>(data);
}
GPUTensor::GPUTensor(const Shape& shape, af::array&& data, dtype type)
    : shape_(shape), type_(type) {
    if (shape.size() != data.elements()) {
        throw std::invalid_argument("Data size does not match shape");
    }
    data_ = std::make_shared<af::array>(std::move(data));
}

template <typename T>
GPUTensor::GPUTensor(const Shape& shape, const std::vector<T>& data)
    : shape_(shape), type_(dtype_trait<T>::value) {
    if (shape.size() != data.size()) {
        throw std::invalid_argument("Data size does not match shape");
    }
    std::vector<dim_t> af_dims(shape.getDimensions().begin(), shape.getDimensions().end());
    data_ = std::make_shared<af::array>(af::array(af_dims.size(), af_dims.data(), data.data()));
}

template <typename T>
GPUTensor::GPUTensor(const Shape& shape, const T* data, size_t num_elements)
    : shape_(shape), type_(dtype_trait<T>::value) {
    if (shape.size() != num_elements) {
        throw std::invalid_argument("Data size does not match shape");
    }
    std::vector<dim_t> af_dims(shape.getDimensions().begin(), shape.getDimensions().end());
    data_ = std::make_shared<af::array>(af::array(af_dims.size(), af_dims.data(), data));
}

GPUTensor::GPUTensor(const GPUTensor& other)
    : shape_(other.shape_), type_(other.type_), data_(std::make_shared<af::array>(*other.data_)) {}

GPUTensor::GPUTensor(GPUTensor&& other) noexcept
    : shape_(std::move(other.shape_)), type_(other.type_), data_(std::move(other.data_)) {}

GPUTensor& GPUTensor::operator=(const GPUTensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        type_ = other.type_;
        data_ = std::make_shared<af::array>(*other.data_);
    }
    return *this;
}

GPUTensor& GPUTensor::operator=(GPUTensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        type_ = other.type_;
        data_ = std::move(other.data_);
    }
    return *this;
}

GPUTensor::GPUTensor(const GPUTensor& other)
    : shape_(other.shape_), type_(other.type_), data_(std::make_shared<af::array>(*other.data_)) {}

GPUTensor::GPUTensor(GPUTensor&& other) noexcept
    : shape_(std::move(other.shape_)), type_(other.type_), data_(std::move(other.data_)) {
        other.data_ = nullptr;
        other.shape_ = Shape();
        other.type_ = dtype::Undefined;
    }

GPUTensor& GPUTensor::operator=(const GPUTensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        type_ = other.type_;
        data_ = std::make_shared<af::array>(*other.data_);
    }
    return *this;
}

GPUTensor& GPUTensor::operator=(GPUTensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        type_ = other.type_;
        data_ = std::move(other.data_);
    }
    return *this;
}

void* GPUTensor::data() {
    return data_.get();
}

const void* GPUTensor::data() const {
    return data_.get();
}

const Shape& GPUTensor::shape() const {
    return shape_;
}

const std::vector<size_t>& GPUTensor::stride() const {
    return shape_.getStride();
}

size_t GPUTensor::size() const {
    return shape_.size();
}

dtype GPUTensor::type() const {
    return type_;
}

Tensor GPUTensor::at(const std::vector<size_t>& indices) const {
    return Tensor(std::make_unique<GPUTensor>(shape_, (*data_)(computeFlatIndex(shape_, indices)), type_));
}

Tensor GPUTensor::at(size_t index) const {
    return Tensor(std::make_unique<GPUTensor>(shape_, (*data_)(index), type_));
}

void GPUTensor::set(const std::vector<size_t>& indices, const DataItem& value) {
    if (indices.size() != shape_.rank()) {
        throw std::invalid_argument("Number of indices must match tensor rank");
    }

    size_t flatIndex = computeFlatIndex(shape_, indices);
    void* dest = (*data_).device<void>();
    convert_dtype(static_cast<void*>(&(*data_)(flatIndex)), value.data, type_, value.type);

    af::sync();
}

void GPUTensor::set(size_t index, const DataItem& value) {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index exceeds tensor size");
    }

    void* dest = (*data_).device<void>();
    convert_dtype(static_cast<void*>(&(*data_)(index)), value.data, type_, value.type);
    af::sync();
}

Tensor GPUTensor::slice(const std::vector<std::pair<size_t, size_t>>& ranges) const {
    if (ranges.size() > 4) {
        throw std::invalid_argument("ArrayFire only supports up to 4 dimensions");
    }

    // Convert ranges to ArrayFire sequences
    std::vector<af::seq> seqs;
    std::vector<int> newShape;

    for (const auto& range : ranges) {
        seqs.emplace_back(static_cast<double>(range.first), static_cast<double>(range.second), 1.0);
        newShape.push_back(static_cast<int>(range.second - range.first + 1));
    }

    // Pad the seqs with spans to support up to 4 dimensions
    seqs.resize(4, af::span);

    // Perform slicing directly using the seqs vector
    af::array sliced = (*data_)(seqs[0], seqs[1], seqs[2], seqs[3]);

    // Return the new sliced tensor
    return Tensor(std::make_unique<GPUTensor>(Shape(newShape), sliced, type_));
}


void GPUTensor::addInPlace(const Tensor& other) {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    *data_ += *otherGPU.data_;
}

void GPUTensor::subtractInPlace(const Tensor& other) {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    *data_ -= *otherGPU.data_;
}

void GPUTensor::multiplyInPlace(const Tensor& other) {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    *data_ *= *otherGPU.data_;
}

void GPUTensor::divideInPlace(const Tensor& other) {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    *data_ /= *otherGPU.data_;
}

void GPUTensor::addScalarInPlace(double scalar) {
    *data_ += scalar;
}

void GPUTensor::subtractScalarInPlace(double scalar) {
    *data_ -= scalar;
}

void GPUTensor::multiplyScalarInPlace(double scalar) {
    *data_ *= scalar;
}

void GPUTensor::divideScalarInPlace(double scalar) {
    *data_ /= scalar;
}

bool GPUTensor::equal(const Tensor& other) const {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    if (shape_ != otherGPU.shape_) {
        return false;
    }
    return af::allTrue<bool>((*data_ == *otherGPU.data_));
}

bool GPUTensor::greaterThan(const Tensor& other) const {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    if (shape_ != otherGPU.shape_) {
        return false;
    }
    return af::allTrue<bool>((*data_ > *otherGPU.data_));
}

bool GPUTensor::lessThan(const Tensor& other) const {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    if (shape_ != otherGPU.shape_) {
        return false;
    }
    return af::allTrue<bool>((*data_ < *otherGPU.data_));
}

std::string GPUTensor::toString() {
    return af::toString("Tensor: ", (*data_));
}

std::string GPUTensor::toDataString() {
    return af::toString("Tensor: ", (*data_));
}

void GPUTensor::fill(const DataItem& value) {
    *data_ = af::constant(value, data_->dims(), data_->type());
}

void GPUTensor::reshape(const Shape& newShape) {
    if (shape_.size() != newShape.size()) {
        throw std::runtime_error("Number of elements must remain constant during reshape");
    }
    shape_ = newShape;
}

std::unique_ptr<TensorAdapter> GPUTensor::clone() const {
    return std::make_unique<GPUTensor>(*this);
}

TensorBackend& GPUTensor::backend() const {
    static GPUTensorBackend backend;
    return backend;
}

double GPUTensor::getValueAsDouble(size_t index) const {
    return utils::getElementAsDouble(*data_, index, type_);
}

void GPUTensor::setValueFromDouble(size_t index, double value) {
    (*data_)(index) = value;
}

}
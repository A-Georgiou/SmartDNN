

#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensor.hpp"
#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/Shape/ShapeOperations.hpp"
#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.hpp"
#include "smart_dnn/Shape/Shape.hpp"
#include <typeindex>
#include <memory>
#include <any>

namespace sdnn {

GPUTensor::~GPUTensor() {}

GPUTensor::GPUTensor(const Shape& shape, dtype type)
    : shape_(shape), type_(type) {
    std::vector<dim_t> af_dims(shape.getDimensions().begin(), shape.getDimensions().end());
    
    // Convert the vector of dimensions to an af::dim4 object
    af::dim4 dims(af_dims.size() > 0 ? af_dims[0] : 1,
                  af_dims.size() > 1 ? af_dims[1] : 1,
                  af_dims.size() > 2 ? af_dims[2] : 1,
                  af_dims.size() > 3 ? af_dims[3] : 1);

    // Use the af::dim4 object in the af::array constructor
    data_ = std::make_shared<af::array>(dims, utils::sdnnToAfType(type));
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

Tensor GPUTensor::at(const std::vector<size_t>& indices) const {
    size_t colMajorIndex = getFlatIndex(computeFlatIndex(shape_, indices));
    return Tensor(std::make_unique<GPUTensor>(shape_, (*data_)(colMajorIndex), type_));
}

Tensor GPUTensor::at(size_t index) const {
    size_t colMajorIndex = getFlatIndex(index);
    return Tensor(std::make_unique<GPUTensor>(shape_, (*data_)(colMajorIndex), type_));
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


void GPUTensor::add(const Tensor& other) {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    *data_ += *otherGPU.data_;
}

void GPUTensor::sub(const Tensor& other) {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    *data_ -= *otherGPU.data_;
}

void GPUTensor::mul(const Tensor& other) {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    *data_ *= *otherGPU.data_;
}

void GPUTensor::div(const Tensor& other) {
    const GPUTensor& otherGPU = dynamic_cast<const GPUTensor&>(*other.tensorImpl_);
    *data_ /= *otherGPU.data_;
}

#define IMPLEMENT_TYPE_SPECIFIC_OPS(TYPE) \
    void GPUTensor::addScalar(TYPE scalar) { \
        *data_ += scalar;  \
    } \
    void GPUTensor::subScalar(TYPE scalar) { \
        *data_ -= scalar;  \
    } \
    void GPUTensor::mulScalar(TYPE scalar) { \
        *data_ *= scalar;  \
    } \
    void GPUTensor::divScalar(TYPE scalar) { \
        if (scalar == 0) throw std::runtime_error("Division by zero");  \
        *data_ /= scalar;  \
    } \
    void GPUTensor::set(size_t index, TYPE value) { \
        size_t colMajorIndex = getFlatIndex(index); \
        (*data_)(colMajorIndex) = value; \
    } \
    void GPUTensor::set(const std::vector<size_t>& indices, TYPE value) { \
        size_t flatIndex = computeFlatIndex(shape_, indices); \
        (*data_)(flatIndex) = value; \
    } \
    void GPUTensor::fill(TYPE value) { \
        af::dim4 dims = utils::shapeToAfDim(shape_); \
        af::array result = af::constant(value, dims, utils::sdnnToAfType(type_)); \
        *data_ = result; \
    } \
    void GPUTensor::getValueAsType(size_t index, TYPE& value) const { \
        size_t flat_index = getFlatIndex(index); \
        if (std::is_same<TYPE, bool>::value) { \
            throw std::runtime_error("ArrayFire does not support bool scalar extraction on this platform."); \
        } else { \
            /* ArrayFire only reliably supports float scalar extraction on this platform */ \
            float temp_value = (*data_)(flat_index).scalar<float>(); \
            value = static_cast<TYPE>(temp_value); \
        } \
    } \

IMPLEMENT_TYPE_SPECIFIC_OPS(bool)
IMPLEMENT_TYPE_SPECIFIC_OPS(int)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned int)
IMPLEMENT_TYPE_SPECIFIC_OPS(long)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned long)
IMPLEMENT_TYPE_SPECIFIC_OPS(long long)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned long long)
IMPLEMENT_TYPE_SPECIFIC_OPS(float)
IMPLEMENT_TYPE_SPECIFIC_OPS(double)
IMPLEMENT_TYPE_SPECIFIC_OPS(char)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned char)
IMPLEMENT_TYPE_SPECIFIC_OPS(short)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned short)

#undef IMPLEMENT_TYPE_SPECIFIC_OPS

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
    return "Shape: " + shape_.toString() + "\n" + af::toString("Tensor: ", (*data_));
}

std::string GPUTensor::toDataString() {
    return af::toString("Tensor: ", (*data_));
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
    return utils::getElementAsDouble(*data_, index);
}

void GPUTensor::setValueFromDouble(size_t index, double value) {
    (*data_)(index) = value;
}

size_t GPUTensor::getFlatIndex(size_t index) const {
    size_t output = rowWiseToColumnMajorIndex(index, shape_);
    return output;
}

size_t GPUTensor::rowMajorToColumnMajorIndex(const std::vector<size_t>& indices, const Shape& shape) const {
    size_t columnMajorIndex = 0;
    size_t stride = 1;

    for (size_t i = 0; i < indices.size(); ++i) {
        columnMajorIndex += indices[i] * stride;
        stride *= shape.getDimensions()[i];
    }

    return columnMajorIndex;
}

size_t GPUTensor::rowWiseToColumnMajorIndex(size_t rowWiseIndex, const Shape& shape) const {
    std::vector<int> dims = shape_.getDimensions();
    size_t numDims = dims.size();
    
    if (numDims < 2) {
        return rowWiseIndex;
    }

    std::vector<size_t> positions(numDims);
    size_t remaining = rowWiseIndex;
    for (int i = numDims - 1; i >= 0; --i) {
        positions[i] = remaining % dims[i];
        remaining /= dims[i];
    }

    size_t columnMajorIndex = 0;
    size_t stride = 1;
    for (size_t i = 0; i < numDims; ++i) {
        columnMajorIndex += positions[i] * stride;
        stride *= dims[i];
    }

    return columnMajorIndex;
}

}
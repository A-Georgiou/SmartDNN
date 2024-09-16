

#include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"
#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensorBackend.hpp"
#include "smart_dnn/shape/Shape.hpp"
#include <typeindex>
#include <memory>

namespace sdnn {

CPUTensor::~CPUTensor() {
    (*data_).clear();
}

CPUTensor::CPUTensor(const Shape& shape, dtype type)
    : shape_(shape), type_(type), data_(std::make_shared<std::vector<char>>()) {
    allocateMemory(shape.size() * dtype_size(type));
}

CPUTensor::CPUTensor(const Shape& shape, const void* data, dtype type)
    : shape_(shape), type_(type), data_(std::make_shared<std::vector<char>>()) {
    size_t total_elements = shape.size();
    size_t type_size = dtype_size(type);
    size_t total_size = total_elements * type_size;
    
    allocateMemory(total_size);
    
    if (data != nullptr) {
        applyTypedOperationHelper(type, [this, data, total_elements](auto dummy) {
            using T = decltype(dummy);
            const T* typed_data = static_cast<const T*>(data);
            for (size_t i = 0; i < total_elements; ++i) {
                writeElement<T>(this->data_->data(), i, typed_data[i]);
            }
        });
    } else {
        std::memset(data_->data(), 0, total_size);
    }
}

CPUTensor::CPUTensor(const CPUTensor& other)
    : shape_(other.shape_), type_(other.type_), data_(other.data_), indexMap_(other.indexMap_) {
}

CPUTensor::CPUTensor(CPUTensor&& other) noexcept {
    shape_ = std::move(other.shape_);
    type_ = std::move(other.type_);
    data_ = std::move(other.data_);
    indexMap_ = std::move(other.indexMap_);
}


CPUTensor& CPUTensor::operator=(const CPUTensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        type_ = other.type_;
        data_ = other.data_; // Share the data
        indexMap_ = other.indexMap_;
    }
    return *this;
}

CPUTensor& CPUTensor::operator=(CPUTensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        type_ = std::move(other.type_);
        data_ = std::move(other.data_);
        indexMap_ = std::move(other.indexMap_);
    }
    return *this;
}

Tensor CPUTensor::operator[](size_t index) {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return Tensor(at(index));
}

const Tensor CPUTensor::operator[](size_t index) const {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return Tensor(at(index));
}

void CPUTensor::set(const std::vector<size_t>& indices, const double& value) {
    size_t flatIndex = computeFlatIndex(shape_, indices);
    setValueFromDouble(flatIndex, value);
}

void CPUTensor::set(size_t index, const double& value){
    setValueFromDouble(index, value);
}

// Suggested improvements
Tensor CPUTensor::at(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.rank()) {
        throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
    }
    size_t flatIndex = computeFlatIndex(shape_, indices);
    return at(flatIndex);
}

Tensor CPUTensor::at(size_t index) const {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return Tensor(std::make_unique<CPUTensor>(Shape({1}), data_, std::vector<size_t>{index}, type_));
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

void CPUTensor::apply(const std::function<void(double&)>& func) {
        applyTypedOperation([&](auto* type_ptr) {
            using T = std::remove_pointer_t<decltype(type_ptr)>;
            T* data = this->typedData<T>();
            for (size_t i = 0; i < this->size(); ++i) {
                double value = static_cast<double>(data[i]);
                func(value);
                data[i] = static_cast<T>(value);
            }
        });
    }

std::unique_ptr<TensorAdapter> CPUTensor::clone() const {
    auto newTensor = std::make_unique<CPUTensor>(shape_, type_);
    newTensor->data_ = std::make_shared<std::vector<char>>(*data_);
    return newTensor;
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

CPUTensor CPUTensor::subView(const std::vector<size_t>& indices) const {
    std::vector<size_t> newIndexMap;
    for (const auto& index : indices) {
        newIndexMap.push_back(index);
    }

    // The new shape will be 1-dimensional, equal to the number of selected indices
    Shape newShape({static_cast<int>(indices.size())});

    return CPUTensor(newShape, data_, std::move(newIndexMap), type_);
}

// Safer allocation.
inline void CPUTensor::allocateMemory(size_t size) {
    if (!data_) {
        data_ = std::make_shared<std::vector<char>>(size);
    } else {
        data_->resize(size);
    }
}

TensorBackend& CPUTensor::backend() const {
    static CPUTensorBackend backend;
    return backend;
}

double CPUTensor::getValueAsDouble(size_t index) const {
    if (!indexMap_.empty()) {
        // If indexMap_ is not empty, use it to access the correct element
        if (index >= indexMap_.size()) {
            throw std::out_of_range("Index out of range");
        }
        index = indexMap_[index]; // Map the index to the flat data index
    }

    double result = 0.0;

    applyTypedOperationHelper(type_, [this, index, &result](auto dummy) {
        using T = decltype(dummy);
        const T* data = reinterpret_cast<const T*>((*data_).data());
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
        T* data = reinterpret_cast<T*>((*data_).data());
        data[index] = static_cast<T>(value);
    });
}


} // namespace sdnn
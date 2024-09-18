

#include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"
#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensorBackend.hpp"
#include "smart_dnn/shape/Shape.hpp"
#include <typeindex>
#include <memory>
#include <any>

namespace sdnn {

CPUTensor::~CPUTensor() {
    (*data_).clear();
}

CPUTensor::CPUTensor(const Shape& shape, dtype type)
    : shape_(shape), type_(type) {
    allocateMemory();
}

CPUTensor::CPUTensor(const CPUTensor& other)
    : shape_(other.shape_), type_(other.type_), data_(other.data_) {
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
        data_ = other.data_; // Share the data
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

Tensor CPUTensor::operator[](size_t index) {
    if (index >= shape_[0]) {
        throw std::out_of_range("Index out of range");
    }
    
    std::vector<int> newDims = shape_.getDimensions();
    newDims.erase(newDims.begin());

    Shape newShape(newDims);
    
    std::optional<TensorIndex> newIndex;
    if (index_) {
        newIndex = index_->subIndex(index);
    } else {
        size_t offset = index * shape_.size() / shape_[0];
        newIndex = TensorIndex(newShape, newShape.getStride(), offset);
    }
    
    return Tensor(std::make_unique<CPUTensor>(
        newShape,
        data_,
        type_,
        newIndex
    ));
}

const Tensor CPUTensor::operator[](size_t index) const {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return at(index);
}

void CPUTensor::set(const std::vector<size_t>& indices, const DataItem& value) {
    size_t flatIndex = index_ ? index_->flattenIndex(indices) : computeFlatIndex(shape_, indices);
    set(flatIndex, value);
}

void CPUTensor::set(size_t index, const DataItem& value) {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range");
    }
    size_t byte_offset = index * dtype_size(type_);
    void* dest = data_->data() + byte_offset;
    convert_dtype(dest, value.data, type_, value.type);
}

// Suggested improvements
Tensor CPUTensor::at(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.rank()) {
        throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
    }
    size_t flatIndex = index_ ? index_->flattenIndex(indices) : computeFlatIndex(shape_, indices);
    return at(flatIndex);
}

Tensor CPUTensor::at(size_t index) const {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range");
    }

    // Create a new shape for the sub-tensor (scalar)
    Shape subShape({1});
    
    // Create a new CPUTensor that shares the same data
    auto subTensor = std::make_unique<CPUTensor>(
        subShape,
        data_,  // Share the same data
        type_
    );
    
    return Tensor(std::move(subTensor));
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
    accessData([&](size_t i) {
        applyTypedOperationHelper(type_, [&](auto dummy) {
            using T = std::remove_pointer_t<decltype(dummy)>;
            T* data = reinterpret_cast<T*>(this->data_->data()) + i;
            double value = static_cast<double>(*data);
            func(value);
            *data = static_cast<T>(value);
        });
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
    accessData([&](size_t i) {
        applyTypedOperationHelper(type_, [&](auto dummy) {
            using T = std::remove_pointer_t<decltype(dummy)>;
            const T* data = reinterpret_cast<const T*>(this->data_->data()) + i;
            if (i > 0) ss << " ";
            ss << *data;
        });
    });
    return ss.str();
}

void CPUTensor::fill(const DataItem& value) {
    for (size_t i = 0; i < shape_.size(); ++i) {
        set(i, value);
    }
}

CPUTensor CPUTensor::subView(const std::vector<size_t>& indices) const {
    std::vector<size_t> newIndexMap;
    for (const auto& index : indices) {
        newIndexMap.push_back(index);
    }

    // The new shape will be 1-dimensional, equal to the number of selected indices
    Shape newShape({static_cast<int>(indices.size())});

    return CPUTensor(newShape, data_, type_);
}

// Safer allocation.
inline void CPUTensor::allocateMemory() {
    size_t size = shape_.size() * dtype_size(type_);
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
    double result = 0.0;
    accessData([&](size_t i) {
        if (i == index) {
            applyTypedOperationHelper(type_, [&](auto dummy) {
                using T = decltype(dummy);
                const T* data = reinterpret_cast<const T*>(data_->data()) + i;
                result = static_cast<double>(*data);
            });
        }
    });
    return result;
}

void CPUTensor::setValueFromDouble(size_t index, double value) {
    accessData([&](size_t i) {
        if (i == index) {
            applyTypedOperationHelper(type_, [&](auto dummy) {
                using T = decltype(dummy);
                T* data = reinterpret_cast<T*>(data_->data()) + i;
                *data = static_cast<T>(value);
            });
        }
    });
}

void CPUTensor::getValueAsType(size_t index, const DataItem& data) const {
    size_t getPosition = index * dtype_size(type_);
    convert_dtype(data.data, data_->data() + getPosition, data.type, type_);
}


} // namespace sdnn
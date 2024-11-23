

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

CPUTensor::~CPUTensor() {}

CPUTensor::CPUTensor(const Shape& shape, dtype type)
    : shape_(shape), type_(type) {
    allocateMemory();
}

CPUTensor::CPUTensor(const CPUTensor& other)
    : shape_(other.shape_), type_(other.type_), data_(other.data_), index_(other.index_) {
}

CPUTensor::CPUTensor(CPUTensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        type_ = std::move(other.type_);
        data_ = std::move(other.data_);
        index_ = std::move(other.index_);
    }
    other.shape_ = Shape({1});
    other.type_ = type_;
    other.data_ = nullptr;
    other.index_ = std::nullopt;
}


CPUTensor& CPUTensor::operator=(const CPUTensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        type_ = other.type_;
        data_ = other.data_; // Share the data
        index_ = other.index_;
    }
    return *this;
}

CPUTensor& CPUTensor::operator=(CPUTensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        type_ = std::move(other.type_);
        data_ = std::move(other.data_);
        index_ = std::move(other.index_);
    }
    other.shape_ = Shape({1});
    other.type_ = type_;
    other.data_ = nullptr;
    other.index_ = std::nullopt;
    return *this;
}

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

        size_t flatIndex = getFlatIndex(index);
        Shape scalarShape({1});
        
        // Create a new CPUTensor that shares the same data and has scalar shape
        auto scalarTensor = std::make_unique<CPUTensor>(
            scalarShape,
            data_,
            type_,
            TensorIndex(scalarShape, {1}, flatIndex)
        );
        
        return Tensor(std::move(scalarTensor));
    }

Tensor CPUTensor::slice(const std::vector<std::pair<size_t, size_t>>& ranges) const {
    // Ensure the number of ranges matches the rank of the tensor
    if (ranges.size() != shape_.rank()) {
        throw std::invalid_argument("Number of ranges must match the tensor's rank.");
    }

    // Create a new sub TensorIndex for the sliced view
    TensorIndex newIndex = index_ ? index_->slice(ranges) : TensorIndex(shape_).slice(ranges);

    // Compute the new shape for the sliced view
    std::vector<int> newShape;
    for (const auto& range : ranges) {
        newShape.push_back(range.second - range.first);  // Calculate the size of the new dimensions
    }

    // Return a new CPUTensor that shares the same data and has the new TensorIndex and shape
    return Tensor(std::make_unique<CPUTensor>(Shape(newShape), data_, type_, newIndex));
}

void CPUTensor::add(const Tensor& other) {
    elementWiseOperation(other, [](auto& a, const auto& b) { a += b; });
}

void CPUTensor::sub(const Tensor& other) {
    elementWiseOperation(other, [](auto& a, const auto& b) { a -= b; });
}

void CPUTensor::mul(const Tensor& other) {
    elementWiseOperation(other, [](auto& a, const auto& b) { a *= b; });
}

void CPUTensor::div(const Tensor& other) {
    elementWiseOperation(other, [](auto& a, const auto& b) { 
        if (b == 0) throw std::runtime_error("Division by zero");
        a /= b; 
    });
}

#define IMPLEMENT_TYPE_SPECIFIC_OPS(TYPE) \
    void CPUTensor::addScalar(TYPE scalar) { \
        scalarOperation(scalar, [](auto& a, double b) { a += b; });  \
    } \
    void CPUTensor::subScalar(TYPE scalar) { \
        scalarOperation(scalar, [](auto& a, double b) { a -= b; });  \
    } \
    void CPUTensor::mulScalar(TYPE scalar) { \
        scalarOperation(scalar, [](auto& a, double b) { a *= b; });  \
    } \
    void CPUTensor::divScalar(TYPE scalar) { \
        if (scalar == 0) throw std::runtime_error("Division by zero");  \
        scalarOperation(scalar, [](auto& a, double b) { a /= b; });  \
    } \
    void CPUTensor::set(size_t index, TYPE value) { \
        if (index >= shape_.size()) { \
            throw std::out_of_range("Index out of range"); \
        } \
        size_t byte_offset = index * dtype_size(type_); \
        void* dest = data_.get() + byte_offset; \
        dtype from_type = dtype_trait<TYPE>::value; \
        convert_dtype(dest, &value, type_, from_type); \
    } \
    void CPUTensor::set(const std::vector<size_t>& indices, TYPE value) { \
        set(computeFlatIndex(shape_, indices), value); \
    } \
    void CPUTensor::fill(TYPE value) { \
        for (size_t i = 0; i < shape_.size(); ++i) { \
            set(i, value); \
        } \
    } \
    void CPUTensor::getValueAsType(size_t index, TYPE& value) const { \
        if (index >= shape_.size()) { \
            throw std::out_of_range("Index out of range"); \
        } \
        size_t flatIndex = getFlatIndex(index); \
        size_t getPosition = flatIndex * dtype_size(type_); \
        dtype to_type = dtype_trait<TYPE>::value; \
        convert_dtype(&value, data_.get() + getPosition, to_type, type_); \
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
            T* dataElement = reinterpret_cast<T*>(this->data_.get()) + i;
            double value = static_cast<double>(*dataElement);
            func(value);
            *dataElement = static_cast<T>(value);
        });
    });
}

std::unique_ptr<TensorAdapter> CPUTensor::clone() const {
    auto newTensor = std::make_unique<CPUTensor>(shape_, type_);
    std::memcpy(newTensor->data_.get(), data_.get(), shape_.size() * dtype_size(type_));
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
            const T* data = reinterpret_cast<const T*>(this->data_.get()) + i;
            if (i > 0) ss << " ";
            ss << *data;
        });
    });
    return ss.str();
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
        data_ = std::shared_ptr<char[]>(new char[size], std::default_delete<char[]>());
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
                const T* data = reinterpret_cast<const T*>(data_.get()) + i;
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
                T* data = reinterpret_cast<T*>(data_.get()) + i;
                *data = static_cast<T>(value);
            });
        }
    });
}

size_t CPUTensor::getFlatIndex(size_t index) const {
    if (index_) {
        std::vector<size_t> indices = unflattenIndex(index, shape_);
        return index_->flattenIndex(indices);
    }
    return index;
}


} // namespace sdnn


#include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"
#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensorBackend.hpp"
#include <typeindex>

namespace sdnn {

    CPUTensor::CPUTensor(const Shape& shape, dtype type)
        : shape_(shape), data_(std::make_unique<T[]>(shape.size())) {}

    // Constructor with shape and data pointer
    CPUTensor::CPUTensor(const Shape& shape, const T* data, dtype type)
        : shape_(shape), data_(std::make_unique<T[]>(shape.size())) {
        std::copy(data, data + shape.size(), data_.get());
    }

    // Constructor with shape and std::vector
    CPUTensor::CPUTensor(const Shape& shape, const std::vector& data, dtype type)
        : CPUTensor(shape, data.data()) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Data size doesn't match shape size");
        }
    }

    // Constructor with shape and std::initializer_list
    CPUTensor::CPUTensor(const Shape& shape, std::initializer_list data)
        : CPUTensor(shape, data.begin()) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Initializer list size doesn't match shape size");
        }
    }

    // Constructor with shape and fill value (overload for T)
    CPUTensor::CPUTensor(const Shape& shape, T value, dtype type)
        : shape_(shape), data_(std::make_unique<T[]>(shape.size())) {
        std::fill(data_.get(), data_.get() + shape.size(), value);
    }

    // Constructor with shape and fill value (overload for double, converts to T)
    CPUTensor::CPUTensor(const Shape& shape, double value, dtype type)
        : shape_(shape), data_(std::make_unique<T[]>(shape.size())) {
        std::fill(data_.get(), data_.get() + shape.size(), static_cast(value));
    }

    // Copy constructor
    CPUTensor::CPUTensor(const CPUTensor& other)
        : shape_(other.shape_), data_(std::make_unique<T[]>(other.shape_.size())) {
        std::copy(other.data_.get(), other.data_.get() + shape_.size(), data_.get());
    }

    void CPUTensor::data(void* data) {
        std::copy(static_cast<const T*>(data), static_cast<const T*>(data) + shape_.size(), data_.get());
    }

    Tensor CPUTensor::addInPlace(const Tensor& other) {
        if (shape_ != other.shape()) {
            throw std::invalid_argument("Shapes don't match");
        }

        const CPUTensor* otherCPU = dynamic_cast<const CPUTensor*>(other.tensorImpl_.get());
        if (!otherCPU) {
            throw std::invalid_argument("Other tensor is not a CPUTensor of the same type");
        }

        for (size_t i = 0; i < shape_.size(); ++i) {
            set(i, at(i) + otherCPU->at(i));
        }

        return Tensor(std::make_unique<CPUTensor>(*this));
    }

    Tensor CPUTensor::subtractInPlace(const Tensor& other) {
        if (shape_ != other.shape()) {
            throw std::invalid_argument("Shapes don't match");
        }

        const CPUTensor* otherCPU = dynamic_cast<const CPUTensor*>(other.tensorImpl_.get());
        if (!otherCPU) {
            throw std::invalid_argument("Other tensor is not a CPUTensor of the same type");
        }

        for (size_t i = 0; i < shape_.size(); ++i) {
            set(i, at(i) - otherCPU->at(i));
        }

        return Tensor(std::make_unique<CPUTensor>(*this));
    }

    Tensor CPUTensor::multiplyInPlace(const Tensor& other) {
        if (shape_ != other.shape()) {
            throw std::invalid_argument("Shapes don't match");
        }

        const CPUTensor* otherCPU = dynamic_cast<const CPUTensor*>(other.tensorImpl_.get());
        if (!otherCPU) {
            throw std::invalid_argument("Other tensor is not a CPUTensor of the same type");
        }

        for (size_t i = 0; i < shape_.size(); ++i) {
            set(i, at(i)*otherCPU->at(i));
        }

        return Tensor(std::make_unique<CPUTensor>(*this));
    }

    Tensor CPUTensor::divideInPlace(const Tensor& other) {
        if (shape_ != other.shape()) {
            throw std::invalid_argument("Shapes don't match");
        }

        const CPUTensor* otherCPU = dynamic_cast<const CPUTensor*>(other.tensorImpl_.get());
        if (!otherCPU) {
            throw std::invalid_argument("Other tensor is not a CPUTensor of the same type");
        }

        for (size_t i = 0; i < shape_.size(); ++i) {
            set(i, at(i)/otherCPU->at(i));
        }

        return Tensor(std::make_unique<CPUTensor>(*this));
    }

    Tensor CPUTensor::addScalarInPlace(double scalar) {
        for (size_t i = 0; i < shape_.size(); ++i) {
            set(i, at(i) + scalar);
        }

        return Tensor(std::make_unique<CPUTensor>(*this));
    }

    Tensor CPUTensor::subtractScalarInPlace(double scalar) {
        for (size_t i = 0; i < shape_.size(); ++i) {
            set(i, at(i) - scalar);
        }

        return Tensor(std::make_unique<CPUTensor>(*this));
    }

    Tensor CPUTensor::multiplyScalarInPlace(double scalar) {
        for (size_t i = 0; i < shape_.size(); ++i) {
            set(i, at(i)*scalar);
        }

        return Tensor(std::make_unique<CPUTensor>(*this));
    }

    Tensor CPUTensor::divideScalarInPlace(double scalar) {
        for (size_t i = 0; i < shape_.size(); ++i) {
            set(i, at(i)/scalar);
        }

        return Tensor(std::make_unique<CPUTensor>(*this));
    }

    bool CPUTensor::equal(const Tensor& other) const {
        if (shape_ != other.shape()) {
            return false;
        }

        const CPUTensor* otherCPU = dynamic_cast<const CPUTensor*>(other.tensorImpl_.get());
        if (!otherCPU) {
            return false;
        }

        for (size_t i = 0; i < shape_.size(); ++i) {
            if (at(i) != otherCPU->at(i)) {
                return false;
            }
        }

        return true;
    }

    std::string CPUTensor::toString() const {
        std::stringstream ss;
        ss << "CPUTensor<>(" << shape_ << ")\n";
        ss << "Data: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            ss << at(i);
            if (i < shape_.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    std::string CPUTensor::toDataString() const {
        std::stringstream ss;
        ss << "CPUTensor<>(" << shape_ << ")\n";
        ss << "Data: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            ss << at(i);
            if (i < shape_.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    void CPUTensor::fill(const double& value) {
        for (size_t i = 0; i < shape_.size(); ++i) {
            set(i, value);
        }
    }

    // Helper function template to simplify type-based data access
    template <typename T>
    double accessElementAt(const DataHolder* holder, size_t index) {
        auto typedData = dynamic_cast<const TypedDataHolder<T>*>(holder);
        if (!typedData) {
            throw std::runtime_error("Failed to cast to the correct data type");
        }
        if (index >= typedData->getSize()) {
            throw std::out_of_range("Index out of bounds");
        }
        return static_cast<double>(typedData->operator[](index));
    }

    double CPUTensor::at(size_t index) const {
        static const std::unordered_map<std::type_index,
                std::function<double(const DataHolder*, size_t)>> typeHandlers = {
            { typeid(float), accessElementAt<float> },
            { typeid(double), accessElementAt<double> },
            { typeid(int), accessElementAt<int> },
            { typeid(short), accessElementAt<short> },
            { typeid(long), accessElementAt<long> },
            { typeid(unsigned char), accessElementAt<unsigned char> },
            { typeid(unsigned short), accessElementAt<unsigned short> },
            { typeid(unsigned int), accessElementAt<unsigned int> },
            { typeid(unsigned long), accessElementAt<unsigned long> }
        };

        // Get the type of the stored data
        auto type = std::type_index(data_->getType());

        // Find the appropriate handler for the data type
        auto handlerIt = typeHandlers.find(type);
        if(handlerIt == typeHandlers.end()) {
            throw std::runtime_error("Unsupported data type in at() function");
        }

        // Call the handler to access the element
        return (handlerIt->second(data_.get(), index));
    }

    Tensor CPUTensor::at(const std::vector<size_t>& indices) const {
        size_t index = shape_[computeFlatIndex(shape_, indices)];
        return Tensor(std::make_unique<CPUTensor>(Shape({1}), at(index)));
    }

    template <typename T>
    void setElementAt(DataHolder* holder, size_t index, const double& value) {
        auto typedData = dynamic_cast<TypedDataHolder<T>*>(holder);
        if (!typedData) {
            throw std::runtime_error("Failed to cast to the correct data type");
        }
        if (index >= typedData->getSize()) {
            throw std::out_of_range("Index out of bounds");
        }
        typedData->operator[](index) = static_cast<T>(value);
    }

    void CPUTensor::set(size_t index, const double& value) {
        // A map of type_info to functions that handle setting values for each type
        static const std::unordered_map<std::type_index,
            std::function<void(DataHolder*, size_t, const double&)>> typeHandlers = {
            { typeid(float), setElementAt<float> },
            { typeid(double), setElementAt<double> },
            { typeid(int), setElementAt<int> },
            { typeid(short), setElementAt<short> },
            { typeid(long), setElementAt<long> },
            { typeid(unsigned char), setElementAt<unsigned char> },
            { typeid(unsigned short), setElementAt<unsigned short> },
            { typeid(unsigned int), setElementAt<unsigned int> },
            { typeid(unsigned long), setElementAt<unsigned long> }
        };

        // Get the type of the stored data
        auto type = std::type_index(data_->getType());

        // Find the appropriate handler for the data type
        auto handlerIt = typeHandlers.find(type);
        if(handlerIt == typeHandlers.end()) {
            throw std::runtime_error("Unsupported data type in set() function");
        }

        // Call the handler to set the element
        handlerIt->second(data_.get(), index, value);
    }

    void set(const std::vector<size_t>& indices, const double& value) {
        size_t index = shape_[computeFlatIndex(shape_, indices)];
        set(index, value);
    }

    std::unique_ptr<TensorAdapter> CPUTensor::clone() const {
        return std::make_unique<CPUTensor>(*this);
    }

    void CPUTensor::reshape(const Shape& newShape) {
        if (shape_.size() != newShape.size()) {
            throw std::invalid_argument("New shape must have the same number of elements");
        }

        shape_ = newShape;
    }

    TensorBackend& CPUTensor::backend() const { 
        static CPUTensorBackend instance;
        return instance;
    }

    // Allocate memory based on dtype
    void CPUTensor::allocateMemory(dtype type, size_t size) {
        data_ = createDataHolder(type, size);
        type_ = type;
    }

    // Get the size of a dtype in bytes
    size_t CPUTensor::getDtypeSize(dtype type) const {
        switch (type) {
            case dtype::f32: return sizeof(float);
            case dtype::f64: return sizeof(double);
            case dtype::s16: return sizeof(short);
            case dtype::s32: return sizeof(int);
            case dtype::s64: return sizeof(long);
            case dtype::u8:  return sizeof(unsigned char);
            case dtype::u16: return sizeof(unsigned short);
            case dtype::u32: return sizeof(unsigned int);
            case dtype::u64: return sizeof(unsigned long);
            default: throw std::runtime_error("Unsupported dtype");
        }
        }

    // Create a DataHolder based on dtype
    std::unique_ptr<DataHolder> CPUTensor::createDataHolder(dtype type, size_t size) {
        switch (type) {
            case dtype::f32: return std::make_unique<TypedDataHolder<float>>(size);
            case dtype::f64: return std::make_unique<TypedDataHolder<double>>(size);
            case dtype::s16: return std::make_unique<TypedDataHolder<short>>(size);
            case dtype::s32: return std::make_unique<TypedDataHolder<int>>(size);
            case dtype::s64: return std::make_unique<TypedDataHolder<long>>(size);
            case dtype::u8:  return std::make_unique<TypedDataHolder<unsigned char>>(size);
            case dtype::u16: return std::make_unique<TypedDataHolder<unsigned short>>(size);
            case dtype::u32: return std::make_unique<TypedDataHolder<unsigned int>>(size);
            case dtype::u64: return std::make_unique<TypedDataHolder<unsigned long>>(size);
            default: throw std::runtime_error("Unsupported dtype");
        }
    }

    template<typename T>
    bool CPUTensor::isCorrectType() const {
        switch (type_) {
            case dtype::f32: return std::is_same_v<T, float>;
            case dtype::f64: return std::is_same_v<T, double>;
            case dtype::s16: return std::is_same_v<T, short>;
            case dtype::s32: return std::is_same_v<T, int>;
            case dtype::s64: return std::is_same_v<T, long>;
            case dtype::u8:  return std::is_same_v<T, unsigned char>;
            case dtype::u16: return std::is_same_v<T, unsigned short>;
            case dtype::u32: return std::is_same_v<T, unsigned int>;
            case dtype::u64: return std::is_same_v<T, unsigned long>;
            default: return false;
        }
    }


} // namespace sdnn
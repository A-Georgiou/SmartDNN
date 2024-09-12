#ifndef CPU_TENSOR_HPP
#define CPU_TENSOR_HPP

#include <iomanip>
#include <functional>
#include <memory>
#include <vector>
#include <initializer_list>
#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/shape/Shape.hpp"
#include "smart_dnn/tensor/TensorAdapterBase.hpp"

namespace sdnn {

// Base class for holding data of any type
class DataHolder {
public:
    virtual ~DataHolder() = default;

    // Virtual method to clone the data (for copying purposes)
    virtual std::unique_ptr<DataHolder> clone() const = 0;

    virtual const std::type_info& getType() const = 0;

    // Virtual method to return a pointer to the data
    virtual void* getData() = 0;
    virtual const void* getData() const = 0;

    // Virtual method to return the size of the data
    virtual size_t getSize() const = 0;
};

// Templated class for holding our data - This is a migration of how TensorData worked
template<typename T>
class TypedDataHolder : public DataHolder {
public:
    TypedDataHolder(size_t size) : data_(std::make_unique<T[]>(size)), size_(size) {}

    std::unique_ptr<DataHolder> clone() const override {
        auto cloned = std::make_unique<TypedDataHolder<T>>(size_);
        std::copy(data_.get(), data_.get() + size_, cloned->data_.get());
        return cloned;
    }

    const std::type_info& getType() const override {
        return typeid(T);
    }
    
    void* getData() override {
        return static_cast<void*>(data_.get());
    }

    const void* getData() const override {
        return static_cast<void*>(data_.get());
    }

    size_t getSize() const override {
        return size_;
    }

    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[index];
    }

    const T& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[index];
    }

private:
    std::unique_ptr<T[]> data_;
    size_t size_;
};


class CPUTensor : public TensorAdapter {
public:
    CPUTensor() = default;
    ~CPUTensor() override = default;

    // Constructor with shape and dtype
    CPUTensor(const Shape& shape, dtype type = dtype::f32);

    // Constructor with shape, data pointer, and dtype
    CPUTensor(const Shape& shape, const void* data, dtype type);

    // Constructor with shape, std::vector, and dtype
    CPUTensor(const Shape& shape, const std::vector<double>& data, dtype type = dtype::f32);

    // Constructor with shape and fill value (overload for double)
    CPUTensor(const Shape& shape, double value, dtype type = dtype::f32);

    // Copy constructor
    CPUTensor(const CPUTensor& other);

    // Move constructor
    CPUTensor(CPUTensor&& other) noexcept = default;

    // Data access
    void data(void* out) override;
    const Shape& shape() const override { return shape_; };
    size_t size() const override { return shape_.size(); };
    dtype type() const override { return type_; }

    // Basic operations
    Tensor addInPlace(const Tensor& other) override;
    Tensor subtractInPlace(const Tensor& other) override;
    Tensor multiplyInPlace(const Tensor& other) override;
    Tensor divideInPlace(const Tensor& other) override;

    // Comparison operator
    bool equal(const Tensor& other) const override;

    // Scalar operations
    Tensor addScalarInPlace(double scalar) override;
    Tensor subtractScalarInPlace(double scalar) override;
    Tensor multiplyScalarInPlace(double scalar) override;
    Tensor divideScalarInPlace(double scalar) override;

    // String representation
    std::string toString() const override;
    std::string toDataString() const override;

    // Fill the tensor with a given value
    void fill(const double& value) override;

    // Element access
    double CPUTensor::at(size_t index) const;

    Tensor at(const std::vector<size_t>& indices) const override;
    void set(size_t index, const double& value) override;
    void set(const std::vector<size_t>& indices, const double& value) override;

    // Shape operations
    void reshape(const Shape& newShape) override;

    // Memory management
    std::unique_ptr<TensorAdapter> clone() const override;

    // Backend access
    TensorBackend& backend() const override;

private:
    Shape shape_;
    dtype type_;
    std::unique_ptr<DataHolder> data_;

    // Helper function to allocate memory based on dtype
    void allocateMemory(dtype type, size_t size);

    // Helper function to get the size of a dtype in bytes
    size_t getDtypeSize(dtype type) const;

    // Helper function to create a DataHolder based on dtype
    std::unique_ptr<DataHolder> createDataHolder(dtype type, size_t size);

    template <typename T>
    bool isCorrectType() const;
};
} // namespace sdnn

#include "smart_dnn/tensor/Backend/Default/CPUTensor.impl.hpp"

#endif // CPU_TENSOR_HPP

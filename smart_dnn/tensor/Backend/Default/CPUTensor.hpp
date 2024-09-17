#ifndef CPU_TENSOR_HPP
#define CPU_TENSOR_HPP

#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <any>
#include <stdexcept>
#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/shape/Shape.hpp"
#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/Backend/Default/TensorIndex.hpp"

namespace sdnn {

class CPUTensor : public TensorAdapter {
public:
    CPUTensor() = default;
    ~CPUTensor() override;

    // Constructors
    CPUTensor(const Shape& shape, dtype type = dtype::f32);
    CPUTensor(const Shape& shape, const void* data, dtype type = dtype::f32);
    CPUTensor(const Shape& shape, std::shared_ptr<std::vector<char>> sharedData, dtype type)
        : shape_(shape), type_(type), data_(std::move(sharedData)) {}

    CPUTensor(const Shape& shape, std::shared_ptr<std::vector<char>> sharedData, dtype type, std::optional<TensorIndex> index)
        : shape_(shape), type_(type), data_(std::move(sharedData)), index_(std::move(index)) {};

    template <typename T>
    CPUTensor(const Shape& shape, const std::vector<T>& data);

    template <typename T>
    CPUTensor(const Shape& shape, const std::vector<T>& data, dtype type);

    template <typename T>
    CPUTensor(const Shape& shape, const T* data, dtype type);

    template <typename T>
    void initializeData(const T* data, size_t total_elements);

    template <typename T>
    CPUTensor(const Shape& shape, const T data, dtype type);

    template <typename T>
    CPUTensor(const Shape& shape, const T data);

    template <typename T>
    CPUTensor(const Shape& shape, std::initializer_list<T> values, dtype type);

    // Copy and move
    CPUTensor(const CPUTensor& other);
    CPUTensor(CPUTensor&& other) noexcept;
    CPUTensor& operator=(const CPUTensor& other);
    CPUTensor& operator=(CPUTensor&& other) noexcept;

    Tensor operator[](size_t index);
    const Tensor operator[](size_t index) const;

    // Data access
    void* data() override { return (*data_).data(); }
    const void* data() const override { return (*data_).data(); }
    const Shape& shape() const override { return shape_; }
    const std::vector<size_t>& stride() const override { return shape_.getStride(); }
    size_t size() const override { return shape_.size(); }
    dtype type() const override { return type_; }

    Tensor at(const std::vector<size_t>& indices) const override;
    Tensor at(size_t index) const override;
    void set(const std::vector<size_t>& indices, const void* value) override;
    void set(size_t index, const void* value) override;

    template <typename T>
    void setValueAtIndex(size_t index, T&& value);

    // Operations
    void addInPlace(const Tensor& other) override;
    void subtractInPlace(const Tensor& other) override;
    void multiplyInPlace(const Tensor& other) override;
    void divideInPlace(const Tensor& other) override;

    void addScalarInPlace(double scalar) override;
    void subtractScalarInPlace(double scalar) override;
    void multiplyScalarInPlace(double scalar) override;
    void divideScalarInPlace(double scalar) override;

    // Utility functions
    bool equal(const Tensor& other) const override;
    bool greaterThan(const Tensor& other) const override;
    bool lessThan(const Tensor& other) const override;

    std::string toString() override;
    std::string toDataString() override;

    void apply(const std::function<void(double&)>& func) override;

    template<typename T>
    void fill(T value);
    void fill(const double& value) override;
    CPUTensor subView(const std::vector<size_t>& indices) const;

    void reshape(const Shape& newShape) override;
    std::unique_ptr<TensorAdapter> clone() const override;
    TensorBackend& backend() const override;

    template<typename TypedOp>
    void applyTypedOperation(TypedOp op);

    template<typename TypedOp>
    void applyTypedOperation(TypedOp op) const;

    template<typename CompOp>
    bool elementWiseComparison(const Tensor& other, CompOp op) const;

    template<typename Op>
    void elementWiseOperation(const Tensor& other, Op op);

    template<typename Op>
    void scalarOperation(double scalar, Op op);

    template<typename T>
    T* typedData() {
        return safe_cast<std::remove_pointer_t<T>>((*data_).data(), type_);
    }

    template<typename T>
    const T* typedData() const {
        return safe_cast<const std::remove_pointer_t<T>>((*data_).data(), type_);
    }

    double getValueAsDouble(size_t index) const override;
    void setValueFromDouble(size_t index, double value) override;

private:
    Shape shape_;
    dtype type_;
    std::shared_ptr<std::vector<char>> data_;
    std::optional<TensorIndex> index_;

    void allocateMemory(size_t size);

    template<typename TargetType, typename SourceType = double>
    void writeElement(void* buffer, size_t index, SourceType value);

    template <typename T>
    void setValueHelper(size_t flatIndex, T&& value);

    void setValueHelper(size_t flatIndex, const void* value);

    template <typename AccessOp>
    void accessData(AccessOp op) const;
};

} // namespace sdnn

#include "CPUTensor.tpp"

#endif // CPU_TENSOR_HPP
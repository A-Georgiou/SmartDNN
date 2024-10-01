#ifndef GPU_TENSOR_HPP
#define GPU_TENSOR_HPP

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
#include "smart_dnn/tensor/TensorIndex.hpp"
#include "arrayfire.h"

namespace sdnn {

class GPUTensor : public TensorAdapter {
public:
    GPUTensor() = default;
    ~GPUTensor() override;

    // Constructors
    GPUTensor(const Shape& shape, dtype type = dtype::f32);
    GPUTensor(const Shape& shape, const af::array& sharedData, dtype type);
    GPUTensor(const Shape& shape, const af::array& sharedData, dtype type, std::optional<TensorIndex> index);
    GPUTensor(const Shape& shape, af::array&& data, dtype type);

    template <typename T>
    GPUTensor(const Shape& shape, const std::vector<T>& data);

    template <typename T>
    GPUTensor(const Shape& shape, const T* data, size_t num_elements);

    template <typename T>
    GPUTensor(const Shape& shape, T data, dtype type = dtype::f32);

    // Copy and move
    GPUTensor(const GPUTensor& other);
    GPUTensor(GPUTensor&& other) noexcept;
    GPUTensor& operator=(const GPUTensor& other);
    GPUTensor& operator=(GPUTensor&& other) noexcept;

    // Data access
    void* data() override { return data_.get(); }
    const void* data() const override { return data_.get(); }
    const Shape& shape() const override { return shape_; }
    const std::vector<size_t>& stride() const override { return shape_.getStride(); }
    size_t size() const override { return shape_.size(); }
    dtype type() const override { return type_; }

    Tensor at(const std::vector<size_t>& indices) const override;
    Tensor at(size_t index) const override;
    void set(const std::vector<size_t>& indices, const DataItem& value) override;
    void set(size_t index, const DataItem& value) override;
    Tensor slice(const std::vector<std::pair<size_t, size_t>>& ranges) const override;

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

    void fill(const DataItem& value) override;

    void reshape(const Shape& newShape) override;
    std::unique_ptr<TensorAdapter> clone() const override;
    TensorBackend& backend() const override;

    double getValueAsDouble(size_t index) const override;
    void setValueFromDouble(size_t index, double value) override;
    void setValueFromType(size_t index, const DataItem& data) override;
    void getValueAsType(size_t index, const DataItem& data) const override;

private:
    Shape shape_;
    dtype type_;
    std::shared_ptr<af::array> data_;
    std::optional<TensorIndex> index_;
};

} // namespace sdnn

#endif // GPU_TENSOR_HPP
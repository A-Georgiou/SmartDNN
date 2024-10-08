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
#include "smart_dnn/tensor/Backend/ArrayFire/Utils.hpp"
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
    GPUTensor(const Shape& shape, const std::vector<T>& data, dtype type);

    template <typename T>
    GPUTensor(const Shape& shape, const T* data, size_t num_elements);

    template <typename T>
    GPUTensor(const Shape& shape, T data, dtype type);

    template <typename T>
    GPUTensor(const Shape& shape, T data);

    template <typename T>
    GPUTensor(const Shape& shape, std::initializer_list<T> values, dtype type);

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
    Tensor slice(const std::vector<std::pair<size_t, size_t>>& ranges) const override;

    // Operations
    void add(const Tensor& other) override;
    void sub(const Tensor& other) override;
    void mul(const Tensor& other) override;
    void div(const Tensor& other) override;

    #define DECLARE_SCALAR_OPS(TYPE) \
        void addScalar(TYPE scalar) override; \
        void subScalar(TYPE scalar) override; \
        void mulScalar(TYPE scalar) override; \
        void divScalar(TYPE scalar) override; \
        void set(size_t index, TYPE value) override; \
        void set(const std::vector<size_t>& indices, TYPE value) override; \
        void fill(TYPE value) override; \
        void getValueAsType(size_t index, TYPE& value) const override; \

    // Generate scalar operations for various types
    DECLARE_SCALAR_OPS(bool)
    DECLARE_SCALAR_OPS(int)
    DECLARE_SCALAR_OPS(unsigned int)
    DECLARE_SCALAR_OPS(long)
    DECLARE_SCALAR_OPS(unsigned long)
    DECLARE_SCALAR_OPS(long long)
    DECLARE_SCALAR_OPS(unsigned long long)
    DECLARE_SCALAR_OPS(float)
    DECLARE_SCALAR_OPS(double)
    DECLARE_SCALAR_OPS(char)
    DECLARE_SCALAR_OPS(unsigned char)
    DECLARE_SCALAR_OPS(short)
    DECLARE_SCALAR_OPS(unsigned short)

    #undef DECLARE_SCALAR_OPS

    // Utility functions
    bool equal(const Tensor& other) const override;
    bool greaterThan(const Tensor& other) const override;
    bool lessThan(const Tensor& other) const override;

    std::string toString() override;
    std::string toDataString() override;

    void reshape(const Shape& newShape) override;
    std::unique_ptr<TensorAdapter> clone() const override;
    TensorBackend& backend() const override;

    double getValueAsDouble(size_t index) const override;
    void setValueFromDouble(size_t index, double value) override;

    af::array& getArray() { return *data_; }
private:
    Shape shape_;
    dtype type_;
    std::shared_ptr<af::array> data_;
    std::optional<TensorIndex> index_;

    size_t getFlatIndex(size_t index) const;
    size_t rowMajorToColumnMajorIndex(const std::vector<size_t>& indices, const Shape& shape) const;
    size_t rowWiseToColumnMajorIndex(size_t rowWiseIndex, const Shape& shape) const;
};

} // namespace sdnn

#include "GPUTensor.tpp"

#endif // GPU_TENSOR_HPP
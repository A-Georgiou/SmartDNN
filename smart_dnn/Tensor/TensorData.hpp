#ifndef TENSOR_DATA_HPP
#define TENSOR_DATA_HPP

#include "smart_dnn/Shape/Shape.hpp"
#include "smart_dnn/Shape/ShapeOperations.hpp"
#include "smart_dnn/Tensor/DeviceTypes.hpp"
#include <iomanip>
#include <functional>
#include <cassert>
#include <initializer_list>

namespace smart_dnn {

// General template declaration
template <typename T, typename DeviceType>
class TensorData;

// Specialization for CPUDevice
template <typename T>
class TensorData<T, CPUDevice> {
public:
    explicit TensorData(Shape dimensions) noexcept;
    TensorData(Shape dimensions, T value) noexcept;
    TensorData(Shape dimensions, const T* data);
    TensorData(Shape dimensions, T* data) noexcept; // used for slicing.
    TensorData(const TensorData& other);
    TensorData(Shape dimensions, std::initializer_list<T> values);
    TensorData(Shape dimensions, const std::vector<T>& values);
    TensorData(Shape dimensions, std::vector<T>&& values) noexcept;


    TensorData(TensorData&&) noexcept = default;
    TensorData& operator=(const TensorData&);
    TensorData& operator=(TensorData&&) noexcept = default;
    
    bool operator==(const TensorData& other) const;
    bool operator!=(const TensorData& other) const;
    ~TensorData() = default;

    // Accessors
    T* data() noexcept { return data_.get(); }
    const T* data() const noexcept { return data_.get(); }
    const Shape& shape() const noexcept { return shape_; }
    size_t size() const noexcept { return shape_.size(); }
    const std::vector<size_t>& stride() const noexcept { return shape_.getStride(); }

    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    T& at(std::initializer_list<int> indices);
    const T& at(std::initializer_list<int> indices) const;
    T& at(const std::vector<int>& indices);
    const T& at(const std::vector<int>& indices) const;

    // Unchecked in release builds, asserted in debug: data(n, c, h, w).
    template <typename... Indices>
    T& operator()(Indices... indices) noexcept { return data_[offset(indices...)]; }
    template <typename... Indices>
    const T& operator()(Indices... indices) const noexcept { return data_[offset(indices...)]; }

    // Fill the tensor with a given value
    void fill(T value) noexcept { std::fill_n(data_.get(), shape_.size(), value); }

    void reshape(const Shape& newShape) { shape_.reshape(newShape); }
    void reshape(const std::vector<int>& dims) { shape_.reshape(dims); }

    // output the data
    std::string toString() const;
    std::string toDataString() const;

    T* begin() const { return data_.get(); }
    T* end() const { return (data_.get() + shape_.size()); }

private:
    Shape shape_;
    std::unique_ptr<T[]> data_;

    template <typename... Indices>
    size_t offset(Indices... indices) const noexcept {
        assert(sizeof...(Indices) == shape_.rank());
        const size_t* stride = shape_.getStride().data();
        size_t flat = 0, dim = 0;
        ((assert(static_cast<size_t>(indices) < static_cast<size_t>(shape_[dim]) && "index out of bounds"), flat += static_cast<size_t>(indices) * stride[dim++]), ...);
        return flat;
    }
};

} // namespace smart_dnn

#include "TensorDataCPU.impl.hpp"

#endif // TENSOR_DATA_HPP

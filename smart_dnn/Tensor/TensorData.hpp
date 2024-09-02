#ifndef TENSOR_DATA_HPP
#define TENSOR_DATA_HPP

#include "../Shape/Shape.hpp"
#include "DeviceTypes.hpp"

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
    TensorData(const TensorData& other);

    TensorData(TensorData&&) noexcept = default;
    TensorData& operator=(const TensorData&);
    TensorData& operator=(TensorData&&) noexcept = default;
    ~TensorData() = default;

    // Accessors
    T* data() noexcept { return data_.get(); }
    const T* data() const noexcept { return data_.get(); }
    const Shape& shape() const noexcept { return shape_; }
    const size_t size() const noexcept { return shape_.size(); }
    const std::vector<int> stride() const noexcept { return shape_.getStride(); }

    // Fill the tensor with a given value
    void fill(T value) noexcept { std::fill_n(data_.get(), shape_.size(), value); }
    void reshape(const Shape& newShape) { shape_.reshape(newShape); }
    void reshape(const std::vector<int>& dims) { shape_.reshape(dims); }

    // output the data
    std::string toString() const;

    T* begin() const { return data_.get(); }
    T* end() const { return (data_.get() + shape_.size()); }

private:
    Shape shape_;
    std::unique_ptr<T[]> data_;
};

} // namespace smart_dnn

#include "TensorDataCPU.impl.hpp"

#endif // TENSOR_DATA_HPP

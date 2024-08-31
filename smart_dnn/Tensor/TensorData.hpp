#ifndef TENSOR_DATA_HPP
#define TENSOR_DATA_HPP

#include "Shape.hpp"
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

    // Fill the tensor with a given value
    void fill(T value) noexcept {
        std::fill_n(data_.get(), shape_.size(), value);
    }

private:
    Shape shape_;
    std::unique_ptr<T[]> data_;
};

} // namespace smart_dnn

#include "TensorData.impl.hpp"

#endif // TENSOR_DATA_HPP

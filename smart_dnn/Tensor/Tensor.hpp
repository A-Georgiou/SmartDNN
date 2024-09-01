#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "TensorData.hpp"
#include "TensorOperations.hpp"
#include "DeviceTypes.hpp"
#include "../Shape/Shape.hpp"
#include "TensorFactory.hpp"

namespace smart_dnn {

template <typename T, typename DeviceType>
class Tensor {
public:
    Tensor(Shape dimensions) noexcept;
    Tensor(Shape dimensions, T value) noexcept;
    Tensor(const TensorData<T, DeviceType>& data) noexcept;
    Tensor(TensorData<T, DeviceType>&& data) noexcept;
    Tensor(Shape dimensions, const T* dataArray);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept = default;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    Tensor& operator+=(T scalar);
    Tensor& operator-=(T scalar);
    Tensor& operator*=(T scalar);
    Tensor& operator/=(T scalar);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(T scalar) const;
    Tensor operator-(T scalar) const;
    Tensor operator*(T scalar) const;
    Tensor operator/(T scalar) const;

    Tensor operator-() const;

    TensorData<T, DeviceType> getData() const noexcept;
    Shape getShape() const noexcept;
    std::string detailedString() const;

    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    // Static factory operations
    static Tensor ones(Shape dimensions);
    static Tensor zeros(Shape dimensions);
    static Tensor rand(Shape dimensions);

private:
    TensorData<T, DeviceType> data_;
};

} // namespace smart_dnn

#include "Tensor.impl.hpp"

#endif // TENSOR_HPP

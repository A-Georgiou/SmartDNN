#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "TensorData.hpp"
#include "TensorOperations.hpp"
#include "DeviceTypes.hpp"
#include "../Shape/Shape.hpp"
#include "TensorFactory.hpp"
#include <functional>

namespace smart_dnn {

template <typename T=float, typename DeviceType=CPUDevice>
class Tensor {
public:
    Tensor(Shape dimensions) noexcept;
    Tensor(Shape dimensions, T value) noexcept;
    Tensor(const TensorData<T, DeviceType>& data) noexcept;
    Tensor(TensorData<T, DeviceType>&& data) noexcept;
    Tensor(Shape dimensions, const T* dataArray);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    ~Tensor() = default;

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

    // Mathematical operations - return new tensor
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(T scalar) const;
    Tensor operator-(T scalar) const;
    Tensor operator*(T scalar) const;
    Tensor operator/(T scalar) const;

    Tensor sqrt() const;

    Tensor operator-() const;

    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    TensorData<T, DeviceType>& getData() noexcept;
    const TensorData<T, DeviceType>& getData() const noexcept;
    Shape getShape() const noexcept;

    std::string toDetailedString() const;
    std::string toDataString() const;

    void reshape(const Shape& newShape);
    void reshape(const std::vector<int>& dims);

    Tensor& apply(std::function<T(T)> func);

    // Static factory operations on Shape
    static Tensor ones(Shape dimensions);
    static Tensor zeros(Shape dimensions);
    static Tensor rand(Shape dimensions);
    
    // Static factory operations on size_t - 1D tensor
    static Tensor ones(int size);
    static Tensor zeros(int size);
    static Tensor rand(int size);
    static Tensor identity(int size);
    

private:
    TensorData<T, DeviceType> data_;
};

} // namespace smart_dnn

#include "Tensor.impl.hpp"

#endif // TENSOR_HPP

#ifndef TENSOR_ARITHMETIC_HPP
#define TENSOR_ARITHMETIC_HPP

#include "TensorData.hpp"

template <typename T>
class TensorArithmetic {
public:
    virtual TensorData<T>& operator+=(const TensorData<T>& other) = 0;
    virtual TensorData<T>& operator-=(const TensorData<T>& other) = 0;
    virtual TensorData<T>& operator*=(const TensorData<T>& other) = 0;
    virtual TensorData<T>& operator/=(const TensorData<T>& other) = 0;

    virtual TensorData<T>& operator+=(T scalar) = 0;
    virtual TensorData<T>& operator-=(T scalar) = 0;
    virtual TensorData<T>& operator*=(T scalar) = 0;
    virtual TensorData<T>& operator/=(T scalar) = 0;

    virtual TensorData<T> operator+(const TensorData<T>& other) const = 0;
    virtual TensorData<T> operator-(const TensorData<T>& other) const = 0;
    virtual TensorData<T> operator*(const TensorData<T>& other) const = 0;
    virtual TensorData<T> operator/(const TensorData<T>& other) const = 0;

    virtual TensorData<T> operator+(T scalar) const = 0;
    virtual TensorData<T> operator-(T scalar) const = 0;
    virtual TensorData<T> operator*(T scalar) const = 0;
    virtual TensorData<T> operator/(T scalar) const = 0;

    virtual ~TensorArithmetic() = default;
};

#endif // TENSOR_ARITHMETIC_HPP
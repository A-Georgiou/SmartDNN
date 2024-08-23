#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "TensorData.hpp"
#include "TensorArithmetic.hpp"
#include "TensorOperations.hpp"
#include "Shape.hpp"

template <typename T>
class Tensor : public TensorArithmetic<T> {
public:
    explicit Tensor(Shape dimensions) noexcept : data(dimensions) {}
    Tensor(Shape dimensions, T value) noexcept : data(dimensions, value) {}
    Tensor(Shape dimensions, const T* dataArray) : data(dimensions, dataArray) {}

    TensorData<T>& operator+=(const TensorData<T>& other) override {
        TensorOperations<T>::add(data, other);
        return data;
    }
    TensorData<T>& operator-=(const TensorData<T>& other) override {
        TensorOperations<T>::subtract(data, other);
        return data;
    }
    TensorData<T>& operator*=(const TensorData<T>& other) override {
        TensorOperations<T>::multiply(data, other);
        return data;
    }
    TensorData<T>& operator/=(const TensorData<T>& other) override {
        TensorOperations<T>::divide(data, other);
        return data;
    }

private:
    TensorData<T> data;
};

#endif // TENSOR_HPP
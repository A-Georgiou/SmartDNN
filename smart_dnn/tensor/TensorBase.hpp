#ifndef TENSOR_BASE_HPP
#define TENSOR_BASE_HPP

#include <vector>
#include <memory>
#include <string>
#include "smart_dnn/shape/Shape.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/tensor/TensorBackendUtil.hpp"
#include "smart_dnn/tensor/TensorIndex.hpp"

// Forward declare TensorAdapter and TensorBackend
namespace sdnn {

class TensorAdapter;
class TensorBackend;

template<typename... Args>
std::unique_ptr<TensorAdapter> createTensorAdapter(Args&&... args);

class Tensor {
public:
    std::unique_ptr<TensorAdapter> tensorImpl_;

    ~Tensor();
    explicit Tensor(std::unique_ptr<TensorAdapter> tensorImpl);

    Tensor(const TensorAdapter& other);

    template <typename T>
    Tensor(const Shape& shape, const std::vector<T>& data, dtype type);

    template <typename T>
    Tensor(const Shape& shape, const std::vector<T>& data);

    template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
    Tensor(const Shape& shape, const T data, dtype type);

    template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
    Tensor(const Shape& shape, const T data);

    template <typename T>
    Tensor(const Shape& shape, std::initializer_list<T> values);

    template <typename T>
    Tensor(std::initializer_list<T> shape);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    // Core operations (in-place)
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    template <typename T>
    Tensor& operator+=(const T& scalar);

    template <typename T>
    Tensor& operator-=(const T& scalar);

    template <typename T>
    Tensor& operator*=(const T& scalar);

    template <typename T>
    Tensor& operator/=(const T& scalar);

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    Tensor operator[](const std::initializer_list<size_t>& indices);
    Tensor operator[](const std::vector<size_t>& indices);
    Tensor operator[](size_t index);

    Tensor at(const std::vector<size_t>& indices) const;
    Tensor at(size_t index) const;

    const Tensor operator[](size_t index) const;

    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    bool operator<(const Tensor& other) const;
    bool operator>(const Tensor& other) const;
    bool operator<=(const Tensor& other) const;
    bool operator>=(const Tensor& other) const;

    template <typename T>
    void set(size_t index, T data);

    template <typename T>
    void set(const std::vector<size_t>& indices, T data);

    Tensor slice(const std::vector<std::pair<size_t, size_t>>& ranges) const;

    Tensor clone() const;

    const Shape& shape() const noexcept;
    dtype type() const noexcept;
    const TensorBackend& backend() const;

    std::string toString() const;
    std::string toDataString() const;

    template <typename T>
    T at(const std::vector<size_t>& indices) const;

    template <typename T>
    T at(size_t index) const;

    template <typename T>
    void getValueAsType(size_t index, T* out) const;

    template <typename T>
    bool isSameType() const;

    template <typename T>
    const T& getImpl() const;
protected:
    Tensor() = default;
};

template <typename T>
dtype dtypeFromType() {
    return dtype_trait<T>::value;
}

// Free functions for element-wise operations
Tensor add(const Tensor& lhs, const Tensor& rhs);
Tensor sub(const Tensor& lhs, const Tensor& rhs);
Tensor mul(const Tensor& lhs, const Tensor& rhs);
Tensor div(const Tensor& lhs, const Tensor& rhs);

// Free operator overloads
Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);

template <typename T>
Tensor add(const Tensor& tensor, const T& scalar) {
    return tensor.backend().add(tensor, scalar);
}

template <typename T>
Tensor sub(const Tensor& tensor, const T& scalar) {
    return tensor.backend().sub(tensor, scalar);
}

template <typename T>
Tensor mul(const Tensor& tensor, const T& scalar) {
    return tensor.backend().mul(tensor, scalar);
}

template <typename T>
Tensor div(const Tensor& tensor, const T& scalar) {
    return tensor.backend().div(tensor, scalar);
}

template <typename T>
Tensor operator+(const T& scalar, const Tensor& tensor) {
    return tensor.backend().add(tensor, scalar);
}

template <typename T>
Tensor operator-(const T& scalar, const Tensor& tensor) {
    return tensor.backend().scalarSub(scalar, tensor);
}

template <typename T>
Tensor operator*(const T& scalar, const Tensor& tensor) {
    return tensor.backend().mul(tensor, scalar);
}

template <typename T>
Tensor operator/(const T& scalar, const Tensor& tensor) {
    return tensor.backend().scalarDiv(scalar, tensor);
}

template <typename T>
Tensor operator+(const Tensor& tensor, const T& scalar) {
    return tensor.backend().add(tensor, scalar);
}

template <typename T>
Tensor operator-(const Tensor& tensor, const T& scalar) {
    return tensor.backend().sub(tensor, scalar);
}

template <typename T>
Tensor operator*(const Tensor& tensor, const T& scalar) {
    return tensor.backend().mul(tensor, scalar);
}

template <typename T>
Tensor operator/(const Tensor& tensor, const T& scalar) {
    return tensor.backend().div(tensor, scalar);
}

Tensor greaterThan(const Tensor& lhs, const Tensor& rhs);
Tensor lessThan(const Tensor& lhs, const Tensor& rhs);
Tensor greaterThan(const Tensor& lhs, const double& scalar);
Tensor lessThan(const Tensor& lhs, const double& scalar);
Tensor greaterThanEqual(const Tensor& lhs, const Tensor& rhs);
Tensor lessThanEqual(const Tensor& lhs, const Tensor& rhs);
Tensor greaterThanEqual(const Tensor& a, const double& scalar);
Tensor lessThanEqual(const Tensor& a, const double& scalar);

Tensor select(const Tensor& condition, const Tensor& a, const Tensor& b);

// Other operations
Tensor matmul(const Tensor& lhs, const Tensor& rhs);
Tensor transpose(const Tensor& tensor, const std::vector<size_t>& axes = {});
Tensor reshape(const Tensor& tensor, const Shape& newShape);
Tensor sqrt(const Tensor& tensor);
Tensor exp(const Tensor& tensor);
Tensor log(const Tensor& tensor);
Tensor abs(const Tensor& tensor);
Tensor tanh(const Tensor& tensor);
Tensor variance(const Tensor& tensor, const Tensor& meanTensor, const std::vector<size_t>& axes = {});
Tensor reciprocal(const Tensor& tensor, double epsilon = 1e-12);

// Reduction operations
Tensor sum(const Tensor& input, const std::vector<size_t>& axes = {}, bool keepDims = false);
Tensor mean(const Tensor& input, const std::vector<size_t>& axes = {}, bool keepDims = false);
Tensor max(const Tensor& input, const std::vector<size_t>& axes = {}, bool keepDims = false);
Tensor selectMax(const Tensor& input, double min_value);
Tensor selectMax(const Tensor& a, const Tensor& b);
Tensor min(const Tensor& input, const std::vector<size_t>& axes = {}, bool keepDims = false);
Tensor clip(const Tensor& input, const double& min, const double& max);

// Creation functions
Tensor zeros(const Shape& shape, dtype type = dtype::f32);
Tensor ones(const Shape& shape, dtype type = dtype::f32);
Tensor rand(const Shape& shape, dtype type = dtype::f32);
Tensor uniformRand(const Shape& shape, dtype type = dtype::f32);

template <typename T>
Tensor fromVector(const Shape& shape, const std::vector<T>& vec, dtype type) {
    return Tensor(createTensorAdapter(shape, vec.data(), type));
}

template <typename T>
Tensor fromVector(const Shape& shape, const std::vector<T>& vec) {
    return fromVector(shape, vec, dtype_trait<T>::value);
}

template <typename T>
Tensor fromVector(const Shape& shape, const std::initializer_list<T> values, dtype type) {
    std::vector<T> vec(values);
    return Tensor(createTensorAdapter(shape, vec, type));
}

template <typename T>
Tensor fromVector(const Shape& shape, const std::initializer_list<T> values) {
    return fromVector(shape, values, dtype_trait<T>::value);
}  

}; // namespace sdnn

#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "smart_dnn/tensor/TensorBase.tpp"

#endif // TENSOR_BASE_HPP
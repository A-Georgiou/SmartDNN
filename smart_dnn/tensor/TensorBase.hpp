#ifndef TENSOR_BASE_HPP
#define TENSOR_BASE_HPP

#include <vector>
#include <memory>
#include "smart_dnn/shape/Shape.hpp"
#include "smart_dnn/DTypes.hpp"

// Forward declare TensorAdapter and TensorBackend
namespace sdnn {

class TensorAdapter;
class TensorBackend;
class TensorView;

class Tensor {
public:
    std::unique_ptr<TensorAdapter> tensorImpl_;

    ~Tensor();
    explicit Tensor(std::unique_ptr<TensorAdapter> tensorImpl);

    template <typename T>
    Tensor(const Shape& shape, const std::vector<T>& data);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    // Core operations (in-place)
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    // Scalar operations (in-place)
    Tensor& operator+=(const double& scalar);
    Tensor& operator-=(const double& scalar);
    Tensor& operator*=(const double& scalar);
    Tensor& operator/=(const double& scalar);

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    TensorView operator[](const std::initializer_list<size_t>& indices);
    TensorView operator[](const std::vector<size_t>& indices);

    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    template <typename T>
    T at(size_t index) const;
    
    template <typename T>
    T at(const std::vector<size_t>& indices) const;

    void set(size_t index, const double& value);
    void set(const std::vector<size_t>& indices, const double& value);

    const Shape& shape() const noexcept;
    dtype type() const noexcept;
    const TensorBackend& backend() const;

    std::string toString() const;
    std::string toDataString() const;

    template <typename T>
    bool isSameType() const;

    template <typename T>
    const T& getImpl() const  {
        const T* impl = dynamic_cast<const T*>(tensorImpl_.get());
        if (!impl) {
            throw std::runtime_error("Invalid tensor implementation type");
        }
        return *impl;
    }
protected:
    Tensor() = default;
};

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

// Scalar operations
Tensor add(const Tensor& tensor, const double& scalar);
Tensor sub(const Tensor& tensor, const double& scalar);
Tensor mul(const Tensor& tensor, const double& scalar);
Tensor div(const Tensor& tensor, const double& scalar);

// Scalar operator overloads
Tensor operator+(const Tensor& tensor, const double& scalar);
Tensor operator-(const Tensor& tensor, const double& scalar);
Tensor operator*(const Tensor& tensor, const double& scalar);
Tensor operator/(const Tensor& tensor, const double& scalar);

Tensor operator+(const double& scalar, const Tensor& tensor);
Tensor operator-(const double& scalar, const Tensor& tensor);
Tensor operator*(const double& scalar, const Tensor& tensor);
Tensor operator/(const double& scalar, const Tensor& tensor);

// Other operations
Tensor matmul(const Tensor& lhs, const Tensor& rhs);
Tensor transpose(const Tensor& tensor, const std::vector<int>& axes = {});
Tensor reshape(const Tensor& tensor, const Shape& newShape);

// Reduction operations
Tensor sum(const Tensor& input, const std::vector<int>& axes = {}, bool keepDims = false);
Tensor mean(const Tensor& input, const std::vector<int>& axes = {}, bool keepDims = false);

// Creation functions
Tensor zeros(const Shape& shape, dtype type = dtype::f32);
Tensor ones(const Shape& shape, dtype type = dtype::f32);
Tensor rand(const Shape& shape, dtype type = dtype::f32);

Tensor fill(const Shape& shape, dtype type = dtype::f32, const double& fillValue = 0.0f);

}; // namespace sdnn

#endif // TENSOR_BASE_HPP
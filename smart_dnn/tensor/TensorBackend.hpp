// TensorBackend.hpp
#ifndef TENSOR_BACKEND_HPP
#define TENSOR_BACKEND_HPP

#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/shape/Shape.hpp"

namespace sdnn {

class Tensor;

class TensorBackend {
public:
    TensorBackend() = default;
    ~TensorBackend() = default;

    virtual Tensor fill(const Shape& shape, const double& value, dtype type) const = 0;

    // Basic operations
    virtual Tensor add(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor sub(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor mul(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor div(const Tensor& a, const Tensor& b) const = 0;

    virtual Tensor add(const Tensor& a, const double& scalar) const = 0;
    virtual Tensor sub(const Tensor& a, const double& scalar) const = 0;
    virtual Tensor mul(const Tensor& a, const double& scalar) const = 0;
    virtual Tensor div(const Tensor& a, const double& scalar) const = 0;

    // Scalar operations
    virtual Tensor scalarSub(const double& scalar, const Tensor& tensor) const = 0;
    virtual Tensor scalarDiv(const double& scalar, const Tensor& tensor) const = 0;

    // Reduction operations
    virtual Tensor sum(const Tensor& tensor, const std::vector<int>& axes, bool keepDims) const = 0;
    virtual Tensor mean(const Tensor& tensor, const std::vector<int>& axes, bool keepDims) const = 0;

    // Element-wise apply operations
    virtual Tensor apply(const Tensor& tensor, const std::function<void(double&)>& func) const = 0;

    // Linear algebra operations
    virtual Tensor matmul(const Tensor& a, const Tensor& b) const = 0;

    // Shape operations
    virtual Tensor reshape(const Tensor& tensor, const Shape& newShape) const = 0;
    virtual Tensor transpose(const Tensor& tensor, const std::vector<int>& axes) const = 0;

    // Element-wise operations
    virtual Tensor exp(const Tensor& tensor) const = 0;
    virtual Tensor log(const Tensor& tensor) const = 0;
    virtual Tensor power(const Tensor& tensor, double exponent) const = 0;
    virtual Tensor sqrt(const Tensor& tensor) const = 0;
    virtual Tensor abs(const Tensor& tensor) const = 0;
    virtual Tensor negative(const Tensor& tensor) const = 0;

    // Comparison operations
    virtual bool equal(const Tensor& a, const Tensor& b) const = 0;
    virtual bool greaterThan(const Tensor& a, const Tensor& b) const = 0;
    virtual bool greaterThanEqual(const Tensor& a, const Tensor& b) const = 0;
    virtual bool lessThan(const Tensor& a, const Tensor& b) const = 0;
    virtual bool lessThanEqual(const Tensor& a, const Tensor& b) const = 0;

    // Random number generation
    virtual Tensor rand(const Shape& shape, dtype type) const = 0;
    virtual Tensor randn(const Shape& shape, dtype type, double min, double max) const = 0;

    // Tensor generation with Shape values
    virtual Tensor zeros(const Shape& shape, dtype type) const = 0;
    virtual Tensor zeros(int size, dtype type) const = 0;

    virtual Tensor ones(const Shape& shape, dtype type) const = 0;
    virtual Tensor ones(int size, dtype type) const = 0;

    virtual Tensor identity(int size, dtype type) const = 0;
    
    // Backend information
    virtual std::string backendName() const = 0;

    // Utility functions
    virtual void print(const Tensor& tensor) = 0;
};


} // namespace sdnn

#endif // TENSOR_BACKEND_HPP
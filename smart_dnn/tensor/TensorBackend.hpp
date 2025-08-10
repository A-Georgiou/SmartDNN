// TensorBackend.hpp
#ifndef TENSOR_BACKEND_HPP
#define TENSOR_BACKEND_HPP

#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/Shape/Shape.hpp"

namespace sdnn {

class Tensor;

class TensorBackend {
public:
    TensorBackend() = default;
    ~TensorBackend() = default;
    // Basic operations
    virtual Tensor add(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor sub(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor mul(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor div(const Tensor& a, const Tensor& b) const = 0;

    #define DECLARE_SCALAR_OPS(TYPE) \
        virtual Tensor add(const Tensor& a, const TYPE& scalar) const = 0; \
        virtual Tensor sub(const Tensor& a, const TYPE& scalar) const = 0; \
        virtual Tensor mul(const Tensor& a, const TYPE& scalar) const = 0; \
        virtual Tensor div(const Tensor& a, const TYPE& scalar) const = 0; \
        virtual Tensor scalarSub(const TYPE& scalar, const Tensor& tensor) const = 0; \
        virtual Tensor scalarDiv(const TYPE& scalar, const Tensor& tensor) const = 0; \
        virtual Tensor fill(const Shape& shape, const TYPE& fillValue, dtype type) const = 0;

    // Generate scalar operations for various types
    DECLARE_SCALAR_OPS(bool)
    DECLARE_SCALAR_OPS(char)
    DECLARE_SCALAR_OPS(signed char)   // Maps to int8_t
    DECLARE_SCALAR_OPS(unsigned char)
    DECLARE_SCALAR_OPS(short)         // 16-bit integer
    DECLARE_SCALAR_OPS(unsigned short) // 16-bit unsigned integer
    DECLARE_SCALAR_OPS(int)
    DECLARE_SCALAR_OPS(unsigned int)
    DECLARE_SCALAR_OPS(long)          // 64-bit integer on most systems
    DECLARE_SCALAR_OPS(unsigned long) // 64-bit unsigned integer on most systems
    DECLARE_SCALAR_OPS(float)
    DECLARE_SCALAR_OPS(double)

    #undef DECLARE_SCALAR_OPS

    // Reduction operations
    virtual Tensor sum(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const = 0;
    virtual Tensor mean(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const = 0;
    virtual Tensor max(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const = 0;
    virtual Tensor selectMax(const Tensor& tensor, const double& min_value) const = 0;
    virtual Tensor selectMax(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor min(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const = 0;
    virtual Tensor clip(const Tensor& tensor, const double& min, const double& max) const = 0;

    // Element-wise apply operations
    virtual Tensor select(const Tensor& condition, const Tensor& a, const Tensor& b) const = 0;

    // Linear algebra operations
    virtual Tensor matmul(const Tensor& a, const Tensor& b) const = 0;

    // Shape operations
    virtual Tensor reshape(const Tensor& tensor, const Shape& newShape) const = 0;
    virtual Tensor transpose(const Tensor& tensor, const std::vector<size_t>& axes) const = 0;

    // Element-wise operations
    virtual Tensor exp(const Tensor& tensor) const = 0;
    virtual Tensor log(const Tensor& tensor) const = 0;
    virtual Tensor power(const Tensor& tensor, double exponent) const = 0;
    virtual Tensor sqrt(const Tensor& tensor) const = 0;
    virtual Tensor abs(const Tensor& tensor) const = 0;
    virtual Tensor tanh(const Tensor& tensor) const = 0;
    virtual Tensor negative(const Tensor& tensor) const = 0;
    virtual Tensor variance(const Tensor& tensor, const Tensor& meanTensor, const std::vector<size_t>& axes) const = 0;
    virtual Tensor reciprocal(const Tensor& tensor, double epsilon) const = 0;

    // Comparison operations
    virtual bool equal(const Tensor& a, const Tensor& b) const = 0;
    virtual bool greaterThan(const Tensor& a, const Tensor& b) const = 0;
    virtual bool greaterThanEqual(const Tensor& a, const Tensor& b) const = 0;
    virtual bool lessThan(const Tensor& a, const Tensor& b) const = 0;
    virtual bool lessThanEqual(const Tensor& a, const Tensor& b) const = 0;

    virtual Tensor prodGreaterThan(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor prodLessThan(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor prodGreaterThanOrEqual(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor prodLessThanOrEqual(const Tensor& a, const Tensor& b) const = 0;

    virtual Tensor prodGreaterThan(const Tensor& a, const double& scalar) const = 0;
    virtual Tensor prodLessThan(const Tensor& a, const double& scalar) const = 0;
    virtual Tensor prodGreaterThanOrEqual(const Tensor& a, const double& scalar) const = 0;
    virtual Tensor prodLessThanOrEqual(const Tensor& a, const double& scalar) const = 0;

    // Random number generation
    virtual Tensor rand(const Shape& shape, dtype type) const = 0;
    virtual Tensor uniformRand(const Shape& shape, dtype type) const = 0;
    virtual Tensor randn(const Shape& shape, dtype type, float min, float max) const = 0;

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
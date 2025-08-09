#ifndef EIGEN_TENSOR_BACKEND_HPP
#define EIGEN_TENSOR_BACKEND_HPP

#include "smart_dnn/shape/Shape.hpp"
#include "smart_dnn/tensor/TensorBackend.hpp" 
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Core>

namespace sdnn {

class Tensor;

class EigenTensorBackend : public TensorBackend {
public:
    EigenTensorBackend() = default;
    ~EigenTensorBackend();

    // Basic operations
    Tensor add(const Tensor& a, const Tensor& b) const override;
    Tensor sub(const Tensor& a, const Tensor& b) const override;
    Tensor mul(const Tensor& a, const Tensor& b) const override;
    Tensor div(const Tensor& a, const Tensor& b) const override;

    #define DECLARE_SCALAR_OPS(TYPE) \
        Tensor add(const Tensor& a, const TYPE& scalar) const override; \
        Tensor sub(const Tensor& a, const TYPE& scalar) const override; \
        Tensor mul(const Tensor& a, const TYPE& scalar) const override; \
        Tensor div(const Tensor& a, const TYPE& scalar) const override; \
        Tensor scalarSub(const TYPE& scalar, const Tensor& tensor) const override; \
        Tensor scalarDiv(const TYPE& scalar, const Tensor& tensor) const override; \
        Tensor fill(const Shape& shape, const TYPE& fillValue, dtype type) const override;

    // Generate scalar operations for various types
    DECLARE_SCALAR_OPS(bool)
    DECLARE_SCALAR_OPS(int)
    DECLARE_SCALAR_OPS(unsigned int)
    DECLARE_SCALAR_OPS(long)
    DECLARE_SCALAR_OPS(unsigned long)
    DECLARE_SCALAR_OPS(long long)
    DECLARE_SCALAR_OPS(unsigned long long)
    DECLARE_SCALAR_OPS(float)
    DECLARE_SCALAR_OPS(double)
    DECLARE_SCALAR_OPS(char)
    DECLARE_SCALAR_OPS(unsigned char)
    DECLARE_SCALAR_OPS(short)
    DECLARE_SCALAR_OPS(unsigned short)

    #undef DECLARE_SCALAR_OPS

    // Reduction operations
    Tensor sum(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const override;
    Tensor mean(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const override;
    Tensor max(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const override;
    Tensor selectMax(const Tensor& tensor, const double& min_value) const override;
    Tensor selectMax(const Tensor& a, const Tensor& b) const override;
    Tensor min(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const override;
    Tensor clip(const Tensor& tensor, const double& min, const double& max) const override;

    // Element-wise apply operations
    Tensor select(const Tensor& condition, const Tensor& a, const Tensor& b) const override;

    // Linear algebra operations
    Tensor matmul(const Tensor& a, const Tensor& b) const override;

    // Shape operations
    Tensor reshape(const Tensor& tensor, const Shape& newShape) const override;
    Tensor transpose(const Tensor& tensor, const std::vector<size_t>& axes) const override;

    // Element-wise operations
    Tensor exp(const Tensor& tensor) const override;
    Tensor log(const Tensor& tensor) const override;
    Tensor power(const Tensor& tensor, double exponent) const override;
    Tensor sqrt(const Tensor& tensor) const override;
    Tensor abs(const Tensor& tensor) const override;
    Tensor tanh(const Tensor& tensor) const override;
    Tensor negative(const Tensor& tensor) const override;
    Tensor variance(const Tensor& tensor, const Tensor& meanTensor, const std::vector<size_t>& axes) const override;
    Tensor reciprocal(const Tensor& tensor, double epsilon) const override;

    // Comparison operations
    bool equal(const Tensor& a, const Tensor& b) const override;
    bool greaterThan(const Tensor& a, const Tensor& b) const override;
    bool greaterThanEqual(const Tensor& a, const Tensor& b) const override;
    bool lessThan(const Tensor& a, const Tensor& b) const override;
    bool lessThanEqual(const Tensor& a, const Tensor& b) const override;

    Tensor prodGreaterThan(const Tensor& a, const Tensor& b) const override;
    Tensor prodLessThan(const Tensor& a, const Tensor& b) const override;
    Tensor prodGreaterThanOrEqual(const Tensor& a, const Tensor& b) const override;
    Tensor prodLessThanOrEqual(const Tensor& a, const Tensor& b) const override;

    Tensor prodGreaterThan(const Tensor& a, const double& scalar) const override;
    Tensor prodLessThan(const Tensor& a, const double& scalar) const override;
    Tensor prodGreaterThanOrEqual(const Tensor& a, const double& scalar) const override;
    Tensor prodLessThanOrEqual(const Tensor& a, const double& scalar) const override;

    // Random number generation
    Tensor rand(const Shape& shape, dtype type) const override;
    Tensor uniformRand(const Shape& shape, dtype type) const override;
    Tensor randn(const Shape& shape, dtype type, float min, float max) const override;

    // Tensor generation with Shape values
    Tensor zeros(const Shape& shape, dtype type) const override;
    Tensor zeros(int size, dtype type) const override;

    Tensor ones(const Shape& shape, dtype type) const override;
    Tensor ones(int size, dtype type) const override;

    Tensor identity(int size, dtype type) const override;
    
    // Backend information
    std::string backendName() const override;

    // Utility functions
    void print(const Tensor& tensor) override;

private:
    // Helper functions for Eigen operations
    Tensor sumNoAxes(const Tensor& tensor) const;
    Tensor meanNoAxes(const Tensor& tensor) const;
    Tensor minNoAxes(const Tensor& tensor) const;
    Tensor maxNoAxes(const Tensor& tensor) const;
};

} // namespace sdnn

#endif // EIGEN_TENSOR_BACKEND_HPP
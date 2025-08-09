#include <memory>
#include "smart_dnn/tensor/Backend/Eigen/EigenTensorBackend.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"
#include "smart_dnn/tensor/Backend/Default/TemplatedOperations.hpp"
#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/tensor/TensorBackend.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/RandomEngine.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

namespace sdnn {

EigenTensorBackend::~EigenTensorBackend() = default;

// Template functions for Eigen operations - note these need to be at namespace level

// Helper function for element-wise operations using Eigen
template<typename Op>
Tensor eigenElementWiseOp(const Tensor& a, const Tensor& b, Op operation) {
    // For now, handle only same shape tensors (no broadcasting)
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for element-wise operations in Eigen backend");
    }
    
    auto result = std::make_unique<CPUTensor>(a.shape(), a.type());
    const auto& a_cpu = a.getImpl<CPUTensor>();
    const auto& b_cpu = b.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        
        const T* a_data = a_cpu.typedData<T>();
        const T* b_data = b_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = a.shape().size();
        
        // Map to Eigen vectors
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> a_vec(a_data, size);
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> b_vec(b_data, size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
        
        operation(a_vec, b_vec, result_vec);
    });
    
    return Tensor(std::move(result));
}

// Helper function for scalar operations using Eigen
template<typename U, typename Op>
Tensor eigenScalarOp(const Tensor& a, const U& scalar, Op operation) {
    auto result = std::make_unique<CPUTensor>(a.shape(), a.type());
    const auto& a_cpu = a.getImpl<CPUTensor>();

    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* a_data = a_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const T scalar_t = static_cast<T>(scalar);
        const size_t size = a.shape().size();

        // Map to Eigen vectors
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> a_vec(a_data, size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
        
        operation(a_vec, scalar_t, result_vec);
    });

    return Tensor(std::move(result));
}

// Basic operations
Tensor EigenTensorBackend::add(const Tensor& a, const Tensor& b) const {
    return eigenElementWiseOp(a, b, [](const auto& a_vec, const auto& b_vec, auto& result_vec) {
        result_vec = a_vec + b_vec;
    });
}

Tensor EigenTensorBackend::sub(const Tensor& a, const Tensor& b) const {
    return eigenElementWiseOp(a, b, [](const auto& a_vec, const auto& b_vec, auto& result_vec) {
        result_vec = a_vec - b_vec;
    });
}

Tensor EigenTensorBackend::mul(const Tensor& a, const Tensor& b) const {
    return eigenElementWiseOp(a, b, [](const auto& a_vec, const auto& b_vec, auto& result_vec) {
        result_vec = a_vec.cwiseProduct(b_vec);
    });
}

Tensor EigenTensorBackend::div(const Tensor& a, const Tensor& b) const {
    return eigenElementWiseOp(a, b, [](const auto& a_vec, const auto& b_vec, auto& result_vec) {
        result_vec = a_vec.cwiseQuotient(b_vec);
    });
}

// Matrix multiplication using Eigen
Tensor EigenTensorBackend::matmul(const Tensor& a, const Tensor& b) const {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    
    // Only handle 2D matrix multiplication for now
    if (a_shape.rank() != 2 || b_shape.rank() != 2) {
        throw std::invalid_argument("Only 2D matrix multiplication supported in Eigen backend");
    }
    
    if (a_shape[1] != b_shape[0]) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication");
    }
    
    Shape result_shape({a_shape[0], b_shape[1]});
    auto result = std::make_unique<CPUTensor>(result_shape, a.type());
    const auto& a_cpu = a.getImpl<CPUTensor>();
    const auto& b_cpu = b.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        
        const T* a_data = a_cpu.typedData<T>();
        const T* b_data = b_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        
        // Map to Eigen matrices
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a_mat(a_data, a_shape[0], a_shape[1]);
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b_mat(b_data, b_shape[0], b_shape[1]);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> result_mat(result_data, result_shape[0], result_shape[1]);
        
        result_mat = a_mat * b_mat;
    });
    
    return Tensor(std::move(result));
}

// Implementation of scalar operations
#define IMPLEMENT_TYPE_SPECIFIC_OPS(TYPE) \
    Tensor EigenTensorBackend::add(const Tensor& a, const TYPE& scalar) const { \
        return eigenScalarOp(a, scalar, [](const auto& a_vec, auto scalar_t, auto& result_vec) { \
            result_vec = a_vec.array() + scalar_t; \
        }); \
    } \
    Tensor EigenTensorBackend::sub(const Tensor& a, const TYPE& scalar) const { \
        return eigenScalarOp(a, scalar, [](const auto& a_vec, auto scalar_t, auto& result_vec) { \
            result_vec = a_vec.array() - scalar_t; \
        }); \
    } \
    Tensor EigenTensorBackend::mul(const Tensor& a, const TYPE& scalar) const { \
        return eigenScalarOp(a, scalar, [](const auto& a_vec, auto scalar_t, auto& result_vec) { \
            result_vec = a_vec * scalar_t; \
        }); \
    } \
    Tensor EigenTensorBackend::div(const Tensor& a, const TYPE& scalar) const { \
        return eigenScalarOp(a, scalar, [](const auto& a_vec, auto scalar_t, auto& result_vec) { \
            result_vec = a_vec / scalar_t; \
        }); \
    } \
    Tensor EigenTensorBackend::scalarSub(const TYPE& scalar, const Tensor& a) const { \
        return eigenScalarOp(a, scalar, [](const auto& a_vec, auto scalar_t, auto& result_vec) { \
            result_vec = (scalar_t - a_vec.array()); \
        }); \
    } \
    Tensor EigenTensorBackend::scalarDiv(const TYPE& scalar, const Tensor& a) const { \
        return eigenScalarOp(a, scalar, [](const auto& a_vec, auto scalar_t, auto& result_vec) { \
            result_vec = scalar_t / a_vec.array(); \
        }); \
    } \
    Tensor EigenTensorBackend::fill(const Shape& shape, const TYPE& fillValue, dtype type) const { \
        auto result = std::make_unique<CPUTensor>(shape, type); \
        result->applyTypedOperation([&](auto* type_ptr) { \
            using T = std::remove_pointer_t<decltype(type_ptr)>; \
            T* data = result->typedData<T>(); \
            std::fill_n(data, shape.size(), static_cast<T>(fillValue)); \
        }); \
        return Tensor(std::move(result)); \
    }

// Generate scalar operations for various types
IMPLEMENT_TYPE_SPECIFIC_OPS(bool)
IMPLEMENT_TYPE_SPECIFIC_OPS(int)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned int)
IMPLEMENT_TYPE_SPECIFIC_OPS(long)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned long)
IMPLEMENT_TYPE_SPECIFIC_OPS(long long)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned long long)
IMPLEMENT_TYPE_SPECIFIC_OPS(float)
IMPLEMENT_TYPE_SPECIFIC_OPS(double)
IMPLEMENT_TYPE_SPECIFIC_OPS(char)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned char)
IMPLEMENT_TYPE_SPECIFIC_OPS(short)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned short)

#undef IMPLEMENT_TYPE_SPECIFIC_OPS

// For now, implement basic stubs for other operations - they can be improved later
std::string EigenTensorBackend::backendName() const {
    return "Eigen";
}

void EigenTensorBackend::print(const Tensor& tensor) {
    // Use CPU backend printing for now
    std::cout << "Eigen Tensor: ";
    // TODO: Implement proper printing
}

// Stub implementations for other required methods (to be implemented later)
Tensor EigenTensorBackend::sum(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
    throw std::runtime_error("sum operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::mean(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
    throw std::runtime_error("mean operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::max(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
    throw std::runtime_error("max operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::selectMax(const Tensor& tensor, const double& min_value) const {
    throw std::runtime_error("selectMax operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::selectMax(const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("selectMax operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::min(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
    throw std::runtime_error("min operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::clip(const Tensor& tensor, const double& min, const double& max) const {
    throw std::runtime_error("clip operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::select(const Tensor& condition, const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("select operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::reshape(const Tensor& tensor, const Shape& newShape) const {
    throw std::runtime_error("reshape operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::transpose(const Tensor& tensor, const std::vector<size_t>& axes) const {
    throw std::runtime_error("transpose operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::exp(const Tensor& tensor) const {
    throw std::runtime_error("exp operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::log(const Tensor& tensor) const {
    throw std::runtime_error("log operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::power(const Tensor& tensor, double exponent) const {
    throw std::runtime_error("power operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::sqrt(const Tensor& tensor) const {
    throw std::runtime_error("sqrt operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::abs(const Tensor& tensor) const {
    throw std::runtime_error("abs operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::tanh(const Tensor& tensor) const {
    throw std::runtime_error("tanh operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::negative(const Tensor& tensor) const {
    throw std::runtime_error("negative operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::variance(const Tensor& tensor, const Tensor& meanTensor, const std::vector<size_t>& axes) const {
    throw std::runtime_error("variance operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::reciprocal(const Tensor& tensor, double epsilon) const {
    throw std::runtime_error("reciprocal operation not yet implemented in Eigen backend");
}

bool EigenTensorBackend::equal(const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("equal operation not yet implemented in Eigen backend");
}

bool EigenTensorBackend::greaterThan(const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("greaterThan operation not yet implemented in Eigen backend");
}

bool EigenTensorBackend::greaterThanEqual(const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("greaterThanEqual operation not yet implemented in Eigen backend");
}

bool EigenTensorBackend::lessThan(const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("lessThan operation not yet implemented in Eigen backend");
}

bool EigenTensorBackend::lessThanEqual(const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("lessThanEqual operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::prodGreaterThan(const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("prodGreaterThan operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::prodLessThan(const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("prodLessThan operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::prodGreaterThanOrEqual(const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("prodGreaterThanOrEqual operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::prodLessThanOrEqual(const Tensor& a, const Tensor& b) const {
    throw std::runtime_error("prodLessThanOrEqual operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::prodGreaterThan(const Tensor& a, const double& scalar) const {
    throw std::runtime_error("prodGreaterThan (scalar) operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::prodLessThan(const Tensor& a, const double& scalar) const {
    throw std::runtime_error("prodLessThan (scalar) operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::prodGreaterThanOrEqual(const Tensor& a, const double& scalar) const {
    throw std::runtime_error("prodGreaterThanOrEqual (scalar) operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::prodLessThanOrEqual(const Tensor& a, const double& scalar) const {
    throw std::runtime_error("prodLessThanOrEqual (scalar) operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::rand(const Shape& shape, dtype type) const {
    throw std::runtime_error("rand operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::uniformRand(const Shape& shape, dtype type) const {
    throw std::runtime_error("uniformRand operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::randn(const Shape& shape, dtype type, float min, float max) const {
    throw std::runtime_error("randn operation not yet implemented in Eigen backend");
}

// Basic tensor creation operations 
Tensor EigenTensorBackend::zeros(const Shape& shape, dtype type) const {
    auto result = std::make_unique<CPUTensor>(shape, type);
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T* data = result->typedData<T>();
        std::fill_n(data, shape.size(), T(0));
    });
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::zeros(int size, dtype type) const {
    return zeros(Shape({size}), type);
}

Tensor EigenTensorBackend::ones(const Shape& shape, dtype type) const {
    auto result = std::make_unique<CPUTensor>(shape, type);
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T* data = result->typedData<T>();
        std::fill_n(data, shape.size(), T(1));
    });
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::ones(int size, dtype type) const {
    return ones(Shape({size}), type);
}

Tensor EigenTensorBackend::identity(int size, dtype type) const {
    auto result = std::make_unique<CPUTensor>(Shape({size, size}), type);
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T* data = result->typedData<T>();
        
        // Map to Eigen matrix and set identity
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(data, size, size);
        mat.setIdentity();
    });
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::sumNoAxes(const Tensor& tensor) const {
    throw std::runtime_error("sumNoAxes operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::meanNoAxes(const Tensor& tensor) const {
    throw std::runtime_error("meanNoAxes operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::minNoAxes(const Tensor& tensor) const {
    throw std::runtime_error("minNoAxes operation not yet implemented in Eigen backend");
}

Tensor EigenTensorBackend::maxNoAxes(const Tensor& tensor) const {
    throw std::runtime_error("maxNoAxes operation not yet implemented in Eigen backend");
}

} // namespace sdnn
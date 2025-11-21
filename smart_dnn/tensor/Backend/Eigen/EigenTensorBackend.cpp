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
#include <cstdlib>
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
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* a_data = a_cpu.typedData<T>();
        const T* b_data = b_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = a.shape().size();
        
        // Map to Eigen vectors - use const_cast for const data pointers
        // Eigen::Map requires non-const scalar type even for const data
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> a_vec(const_cast<T*>(a_data), size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> b_vec(const_cast<T*>(b_data), size);
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
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        const T* a_data = a_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const T scalar_t = static_cast<T>(scalar);
        const size_t size = a.shape().size();

        // Map to Eigen vectors - use const_cast for const data pointers
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> a_vec(const_cast<T*>(a_data), size);
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
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* a_data = a_cpu.typedData<T>();
        const T* b_data = b_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        
        // Map to Eigen matrices
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a_mat(const_cast<T*>(a_data), a_shape[0], a_shape[1]);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b_mat(const_cast<T*>(b_data), b_shape[0], b_shape[1]);
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
            using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>; \
            T* data = result->typedData<T>(); \
            std::fill_n(data, shape.size(), static_cast<T>(fillValue)); \
        }); \
        return Tensor(std::move(result)); \
    }

// Generate scalar operations for various types
IMPLEMENT_TYPE_SPECIFIC_OPS(bool)
IMPLEMENT_TYPE_SPECIFIC_OPS(char)
IMPLEMENT_TYPE_SPECIFIC_OPS(signed char)     // Maps to int8_t
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned char)
IMPLEMENT_TYPE_SPECIFIC_OPS(short)           // 16-bit integer
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned short)  // 16-bit unsigned integer
IMPLEMENT_TYPE_SPECIFIC_OPS(int)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned int)
IMPLEMENT_TYPE_SPECIFIC_OPS(long)
IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned long)
IMPLEMENT_TYPE_SPECIFIC_OPS(float)
IMPLEMENT_TYPE_SPECIFIC_OPS(double)

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
    if (axes.empty()) {
        return sumNoAxes(tensor);
    }
    
    // For simple case of summing all elements (1D or when axes covers all dimensions)
    if (axes.size() == tensor.shape().rank()) {
        return sumNoAxes(tensor);
    }
    
    // For more complex cases, fall back for now
    throw std::runtime_error("sum with specific axes not yet fully implemented in Eigen backend");
}

Tensor EigenTensorBackend::mean(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
    if (axes.empty()) {
        return meanNoAxes(tensor);
    }
    
    // For simple case of computing mean of all elements
    if (axes.size() == tensor.shape().rank()) {
        return meanNoAxes(tensor);
    }
    
    // For more complex cases, fall back for now
    throw std::runtime_error("mean with specific axes not yet fully implemented in Eigen backend");
}

Tensor EigenTensorBackend::max(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
    if (axes.empty()) {
        return maxNoAxes(tensor);
    }
    
    // For simple case
    if (axes.size() == tensor.shape().rank()) {
        return maxNoAxes(tensor);
    }
    
    throw std::runtime_error("max with specific axes not yet fully implemented in Eigen backend");
}

Tensor EigenTensorBackend::selectMax(const Tensor& tensor, const double& min_value) const {
    auto result = std::make_unique<CPUTensor>(tensor.shape(), tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* input_data = input_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = tensor.shape().size();
        const T min_val = static_cast<T>(min_value);
        
        // Map to Eigen vectors
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(input_data, size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
        
        result_vec = input_vec.cwiseMax(min_val);
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::selectMax(const Tensor& a, const Tensor& b) const {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for selectMax");
    }
    
    auto result = std::make_unique<CPUTensor>(a.shape(), a.type());
    const auto& a_cpu = a.getImpl<CPUTensor>();
    const auto& b_cpu = b.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* a_data = a_cpu.typedData<T>();
        const T* b_data = b_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = a.shape().size();
        
        // Map to Eigen vectors
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> a_vec(a_data, size);
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> b_vec(b_data, size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
        
        result_vec = a_vec.cwiseMax(b_vec);
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::min(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
    if (axes.empty()) {
        return minNoAxes(tensor);
    }
    
    // For simple case
    if (axes.size() == tensor.shape().rank()) {
        return minNoAxes(tensor);
    }
    
    throw std::runtime_error("min with specific axes not yet fully implemented in Eigen backend");
}

Tensor EigenTensorBackend::clip(const Tensor& tensor, const double& min, const double& max) const {
    auto result = std::make_unique<CPUTensor>(tensor.shape(), tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* input_data = input_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = tensor.shape().size();
        const T min_val = static_cast<T>(min);
        const T max_val = static_cast<T>(max);
        
        // Map to Eigen vectors
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(input_data, size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
        
        result_vec = input_vec.cwiseMax(min_val).cwiseMin(max_val);
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::select(const Tensor& condition, const Tensor& a, const Tensor& b) const {
    if (condition.shape() != a.shape() || a.shape() != b.shape()) {
        throw std::invalid_argument("All tensors must have the same shape for select");
    }
    
    auto result = std::make_unique<CPUTensor>(a.shape(), a.type());
    const auto& cond_cpu = condition.getImpl<CPUTensor>();
    const auto& a_cpu = a.getImpl<CPUTensor>();
    const auto& b_cpu = b.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* a_data = a_cpu.typedData<T>();
        const T* b_data = b_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = a.shape().size();
        
        // Get condition data (assume it's stored as bool or numeric)
        cond_cpu.applyTypedOperation([&](auto* cond_type_ptr) {
            using CondT = std::remove_pointer_t<decltype(cond_type_ptr)>;
            const CondT* cond_data = cond_cpu.typedData<CondT>();
            
            // Perform element-wise selection
            for (size_t i = 0; i < size; ++i) {
                result_data[i] = (cond_data[i] != CondT(0)) ? a_data[i] : b_data[i];
            }
        });
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::reshape(const Tensor& tensor, const Shape& newShape) const {
    // Reshape doesn't change the data, just the shape interpretation
    if (tensor.shape().size() != newShape.size()) {
        throw std::invalid_argument("Cannot reshape tensor: total size mismatch");
    }
    
    auto result = std::make_unique<CPUTensor>(newShape, tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* input_data = input_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = tensor.shape().size();
        
        // Simple memory copy since reshape only changes the shape metadata
        std::memcpy(result_data, input_data, size * sizeof(T));
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::transpose(const Tensor& tensor, const std::vector<size_t>& axes) const {
    // For 2D matrices, we can use Eigen's transpose
    if (tensor.shape().rank() == 2 && axes.empty()) {
        auto shape = tensor.shape();
        Shape result_shape({shape[1], shape[0]});
        auto result = std::make_unique<CPUTensor>(result_shape, tensor.type());
        const auto& input_cpu = tensor.getImpl<CPUTensor>();
        
        result->applyTypedOperation([&](auto* type_ptr) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
            
            const T* input_data = input_cpu.typedData<T>();
            T* result_data = result->typedData<T>();
            
            // Map to Eigen matrices
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                input_mat(const_cast<T*>(input_data), shape[0], shape[1]);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                result_mat(result_data, result_shape[0], result_shape[1]);
            
            result_mat = input_mat.transpose();
        });
        
        return Tensor(std::move(result));
    }
    
    // For general case or specified axes, fall back to manual transpose
    throw std::runtime_error("General transpose with arbitrary axes not yet implemented in Eigen backend");
}

// Element-wise mathematical operations using Eigen
Tensor EigenTensorBackend::exp(const Tensor& tensor) const {
    // Check if tensor type is boolean - mathematical operations don't make sense for boolean types
    if (tensor.type() == dtype::b8) {
        throw std::invalid_argument("exp operation is not supported for boolean tensors");
    }
    
    auto result = std::make_unique<CPUTensor>(tensor.shape(), tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        // Use constexpr if to avoid instantiation for bool
        if constexpr (!std::is_same_v<T, bool>) {
            const T* input_data = input_cpu.typedData<T>();
            T* result_data = result->typedData<T>();
            const size_t size = tensor.shape().size();
            
            // Map to Eigen vectors
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(input_data, size);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
            
            result_vec = input_vec.array().exp();
        } else {
            // This should never be reached due to the guard, but we need it for compilation
            throw std::runtime_error("Boolean type should have been caught by guard");
        }
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::sqrt(const Tensor& tensor) const {
    // Check if tensor type is boolean - mathematical operations don't make sense for boolean types
    if (tensor.type() == dtype::b8) {
        throw std::invalid_argument("sqrt operation is not supported for boolean tensors");
    }
    
    auto result = std::make_unique<CPUTensor>(tensor.shape(), tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        // Use constexpr if to avoid instantiation for bool
        if constexpr (!std::is_same_v<T, bool>) {
            const T* input_data = input_cpu.typedData<T>();
            T* result_data = result->typedData<T>();
            const size_t size = tensor.shape().size();
            
            // Map to Eigen vectors
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(input_data, size);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
            
            result_vec = input_vec.array().sqrt();
        } else {
            // This should never be reached due to the guard, but we need it for compilation
            throw std::runtime_error("Boolean type should have been caught by guard");
        }
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::tanh(const Tensor& tensor) const {
    auto result = std::make_unique<CPUTensor>(tensor.shape(), tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* input_data = input_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = tensor.shape().size();
        
        // Map to Eigen vectors
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(input_data, size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
        
        result_vec = input_vec.array().tanh();
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::abs(const Tensor& tensor) const {
    auto result = std::make_unique<CPUTensor>(tensor.shape(), tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* input_data = input_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = tensor.shape().size();
        
        // Map to Eigen vectors
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(input_data, size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
        
        result_vec = input_vec.array().abs();
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::negative(const Tensor& tensor) const {
    auto result = std::make_unique<CPUTensor>(tensor.shape(), tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* input_data = input_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = tensor.shape().size();
        
        // Map to Eigen vectors
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(input_data, size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
        
        result_vec = -input_vec;
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::variance(const Tensor& tensor, const Tensor& meanTensor, const std::vector<size_t>& axes) const {
    // Calculate variance: E[(X - mean)^2] 
    Tensor diff = sub(tensor, meanTensor);
    Tensor squaredDiff = mul(diff, diff);
    Tensor summedSquaredDiff = sum(squaredDiff, axes, false);
    
    // Calculate number of elements being summed over
    float totalElements = 1.0f;
    for (size_t axis : axes) {
        totalElements *= static_cast<float>(tensor.shape()[axis]);
    }
    
    return div(summedSquaredDiff, totalElements);
}

Tensor EigenTensorBackend::reciprocal(const Tensor& tensor, double epsilon) const {
    auto result = std::make_unique<CPUTensor>(tensor.shape(), tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* input_data = input_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = tensor.shape().size();
        const T eps = static_cast<T>(epsilon);
        
        // Map to Eigen vectors
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(input_data, size);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
        
        result_vec = (input_vec.array() + eps).inverse();
    });
    
    return Tensor(std::move(result));
}

bool EigenTensorBackend::equal(const Tensor& a, const Tensor& b) const {
    return a.tensorImpl_->equal(b);
}

bool EigenTensorBackend::greaterThan(const Tensor& a, const Tensor& b) const {
    return a.tensorImpl_->greaterThan(b);
}

bool EigenTensorBackend::greaterThanEqual(const Tensor& a, const Tensor& b) const {
    return a.tensorImpl_->greaterThan(b) || a.tensorImpl_->equal(b);
}

bool EigenTensorBackend::lessThan(const Tensor& a, const Tensor& b) const {
    return a.tensorImpl_->lessThan(b);
}

bool EigenTensorBackend::lessThanEqual(const Tensor& a, const Tensor& b) const {
    return a.tensorImpl_->lessThan(b) || a.tensorImpl_->equal(b);
}

Tensor EigenTensorBackend::prodGreaterThan(const Tensor& a, const Tensor& b) const {
    return eigenElementWiseOp(a, b, [](const auto& a_vec, const auto& b_vec, auto& result_vec) {
        result_vec = (a_vec.array() > b_vec.array()).template cast<typename std::remove_reference<decltype(result_vec)>::type::Scalar>();
    });
}

Tensor EigenTensorBackend::prodLessThan(const Tensor& a, const Tensor& b) const {
    return eigenElementWiseOp(a, b, [](const auto& a_vec, const auto& b_vec, auto& result_vec) {
        result_vec = (a_vec.array() < b_vec.array()).template cast<typename std::remove_reference<decltype(result_vec)>::type::Scalar>();
    });
}

Tensor EigenTensorBackend::prodGreaterThanOrEqual(const Tensor& a, const Tensor& b) const {
    return eigenElementWiseOp(a, b, [](const auto& a_vec, const auto& b_vec, auto& result_vec) {
        result_vec = (a_vec.array() >= b_vec.array()).template cast<typename std::remove_reference<decltype(result_vec)>::type::Scalar>();
    });
}

Tensor EigenTensorBackend::prodLessThanOrEqual(const Tensor& a, const Tensor& b) const {
    return eigenElementWiseOp(a, b, [](const auto& a_vec, const auto& b_vec, auto& result_vec) {
        result_vec = (a_vec.array() <= b_vec.array()).template cast<typename std::remove_reference<decltype(result_vec)>::type::Scalar>();
    });
}

Tensor EigenTensorBackend::prodGreaterThan(const Tensor& a, const double& scalar) const {
    return eigenScalarOp(a, scalar, [](const auto& a_vec, auto scalar_t, auto& result_vec) {
        result_vec = (a_vec.array() > scalar_t).template cast<typename std::remove_reference<decltype(result_vec)>::type::Scalar>();
    });
}

Tensor EigenTensorBackend::prodLessThan(const Tensor& a, const double& scalar) const {
    return eigenScalarOp(a, scalar, [](const auto& a_vec, auto scalar_t, auto& result_vec) {
        result_vec = (a_vec.array() < scalar_t).template cast<typename std::remove_reference<decltype(result_vec)>::type::Scalar>();
    });
}

Tensor EigenTensorBackend::prodGreaterThanOrEqual(const Tensor& a, const double& scalar) const {
    return eigenScalarOp(a, scalar, [](const auto& a_vec, auto scalar_t, auto& result_vec) {
        result_vec = (a_vec.array() >= scalar_t).template cast<typename std::remove_reference<decltype(result_vec)>::type::Scalar>();
    });
}

Tensor EigenTensorBackend::prodLessThanOrEqual(const Tensor& a, const double& scalar) const {
    return eigenScalarOp(a, scalar, [](const auto& a_vec, auto scalar_t, auto& result_vec) {
        result_vec = (a_vec.array() <= scalar_t).template cast<typename std::remove_reference<decltype(result_vec)>::type::Scalar>();
    });
}

Tensor EigenTensorBackend::rand(const Shape& shape, dtype type) const {
    auto result = std::make_unique<CPUTensor>(shape, type);
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        T* result_data = result->typedData<T>();
        const size_t size = shape.size();
        
        // Use Eigen's random functionality
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(result_data, size);
        vec.setRandom(); // Generates values in [-1, 1]
        vec = (vec.array() + T(1)) / T(2); // Scale to [0, 1]
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::uniformRand(const Shape& shape, dtype type) const {
    return rand(shape, type);  // Use same improved implementation
}

Tensor EigenTensorBackend::randn(const Shape& shape, dtype type, float min, float max) const {
    auto result = std::make_unique<CPUTensor>(shape, type);
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        T* result_data = result->typedData<T>();
        const size_t size = shape.size();
        
        // Use Eigen's random functionality  
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(result_data, size);
        vec.setRandom(); // Generates values in [-1, 1]
        
        // Scale to [min, max]
        vec = vec.array() * T((max - min) / 2.0f) + T((max + min) / 2.0f);
    });
    
    return Tensor(std::move(result));
}

// Basic tensor creation operations 
Tensor EigenTensorBackend::zeros(const Shape& shape, dtype type) const {
    auto result = std::make_unique<CPUTensor>(shape, type);
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
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
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
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
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        T* data = result->typedData<T>();
        
        // Map to Eigen matrix and set identity
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(data, size, size);
        mat.setIdentity();
    });
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::sumNoAxes(const Tensor& tensor) const {
    double sum = 0.0;
    const auto& tensor_cpu = tensor.getImpl<CPUTensor>();
    
    tensor_cpu.applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* data = tensor_cpu.typedData<T>();
        const size_t size = tensor.shape().size();
        
        // Map to Eigen vector for efficient summation
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(const_cast<T*>(data), size);
        sum = static_cast<double>(vec.sum());
    });
    
    return Tensor({1}, sum);
}

Tensor EigenTensorBackend::meanNoAxes(const Tensor& tensor) const {
    double mean = 0.0;
    const auto& tensor_cpu = tensor.getImpl<CPUTensor>();
    
    tensor_cpu.applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* data = tensor_cpu.typedData<T>();
        const size_t size = tensor.shape().size();
        
        // Map to Eigen vector for efficient mean computation
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(const_cast<T*>(data), size);
        mean = static_cast<double>(vec.mean());
    });
    
    return Tensor({1}, mean);
}

Tensor EigenTensorBackend::minNoAxes(const Tensor& tensor) const {
    double min_val = 0.0;
    const auto& tensor_cpu = tensor.getImpl<CPUTensor>();
    
    tensor_cpu.applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* data = tensor_cpu.typedData<T>();
        const size_t size = tensor.shape().size();
        
        // Map to Eigen vector for efficient min computation
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(const_cast<T*>(data), size);
        min_val = static_cast<double>(vec.minCoeff());
    });
    
    return Tensor({1}, min_val);
}

Tensor EigenTensorBackend::maxNoAxes(const Tensor& tensor) const {
    double max_val = 0.0;
    const auto& tensor_cpu = tensor.getImpl<CPUTensor>();
    
    tensor_cpu.applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        const T* data = tensor_cpu.typedData<T>();
        const size_t size = tensor.shape().size();
        
        // Map to Eigen vector for efficient max computation
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(const_cast<T*>(data), size);
        max_val = static_cast<double>(vec.maxCoeff());
    });
    
    return Tensor({1}, max_val);
}

Tensor EigenTensorBackend::log(const Tensor& tensor) const {
    // Check if tensor type is boolean - mathematical operations don't make sense for boolean types
    if (tensor.type() == dtype::b8) {
        throw std::invalid_argument("log operation is not supported for boolean tensors");
    }
    
    auto result = std::make_unique<CPUTensor>(tensor.shape(), tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        // Use constexpr if to avoid instantiation for bool
        if constexpr (!std::is_same_v<T, bool>) {
            const T* input_data = input_cpu.typedData<T>();
            T* result_data = result->typedData<T>();
            const size_t size = tensor.shape().size();
            
            // Map to Eigen vectors
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(input_data, size);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
            
            result_vec = input_vec.array().log();
        } else {
            // This should never be reached due to the guard, but we need it for compilation
            throw std::runtime_error("Boolean type should have been caught by guard");
        }
    });
    
    return Tensor(std::move(result));
}

Tensor EigenTensorBackend::power(const Tensor& tensor, double exponent) const {
    // Check if tensor type is boolean - mathematical operations don't make sense for boolean types
    if (tensor.type() == dtype::b8) {
        throw std::invalid_argument("power operation is not supported for boolean tensors");
    }
    
    auto result = std::make_unique<CPUTensor>(tensor.shape(), tensor.type());
    const auto& input_cpu = tensor.getImpl<CPUTensor>();
    
    // Use a conditional template to avoid instantiating pow for bool
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(type_ptr)>>;
        
        // This should never be reached for bool due to the guard above,
        // but we need to handle it at compile time to avoid template errors
        if constexpr (!std::is_same_v<T, bool>) {
            const T* input_data = input_cpu.typedData<T>();
            T* result_data = result->typedData<T>();
            const size_t size = tensor.shape().size();
            
            // Map to Eigen vectors
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> input_vec(input_data, size);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_vec(result_data, size);
            
            result_vec = input_vec.array().pow(static_cast<T>(exponent));
        } else {
            // This should never be reached due to the guard, but we need it for compilation
            throw std::runtime_error("Boolean type should have been caught by guard");
        }
    });
    
    return Tensor(std::move(result));
}

} // namespace sdnn
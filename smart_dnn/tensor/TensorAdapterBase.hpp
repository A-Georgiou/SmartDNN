#ifndef TENSOR_ADAPTER_BASE_HPP
#define TENSOR_ADAPTER_BASE_HPP

#include <vector>
#include <memory>
#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/Shape/Shape.hpp"
#include "smart_dnn/tensor/TensorIndex.hpp"

namespace sdnn {

class Tensor;
class TensorBackend;

class TensorAdapter {
public:
    TensorAdapter() = default;
    virtual ~TensorAdapter();

    // Existing methods that are already virtual
    virtual void* data() = 0;
    virtual const void* data() const = 0;
    virtual const Shape& shape() const = 0;
    virtual size_t size() const = 0;
    virtual const std::vector<size_t>& stride() const = 0;
    virtual dtype type() const = 0;

    // Basic operations
    virtual void add(const Tensor& other) = 0;
    virtual void sub(const Tensor& other) = 0;
    virtual void mul(const Tensor& other) = 0;
    virtual void div(const Tensor& other) = 0;

    // Comparison operator
    virtual bool equal(const Tensor& other) const = 0;
    virtual bool greaterThan(const Tensor& other) const = 0;
    virtual bool lessThan(const Tensor& other) const = 0;

    #define DECLARE_SCALAR_OPS(TYPE) \
        virtual void addScalar(TYPE scalar) = 0; \
        virtual void subScalar(TYPE scalar) = 0; \
        virtual void mulScalar(TYPE scalar) = 0; \
        virtual void divScalar(TYPE scalar) = 0; \
        virtual void set(size_t index, TYPE value) = 0; \
        virtual void set(const std::vector<size_t>& indices, TYPE value) = 0; \
        virtual void fill(TYPE value) = 0; \
        virtual void getValueAsType(size_t index, TYPE& value) const = 0; \

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

    // String representation.
    virtual std::string toString() = 0;
    virtual std::string toDataString() = 0;

    // Element access.
    virtual Tensor at(const std::vector<size_t>& indices) const = 0;
    virtual Tensor at(size_t index) const = 0;

    virtual Tensor slice(const std::vector<std::pair<size_t, size_t>>& ranges) const = 0;

    // Shape operations.
    virtual void reshape(const Shape& newShape) = 0;

    // Memory management
    virtual std::unique_ptr<TensorAdapter> clone() const = 0;

    // Backend access
    virtual TensorBackend& backend() const = 0;

    virtual double getValueAsDouble(size_t index) const = 0;
    virtual void setValueFromDouble(size_t index, double value) = 0;
};

} // namespace sdnn

#endif // TENSOR_ADAPTER_HPP
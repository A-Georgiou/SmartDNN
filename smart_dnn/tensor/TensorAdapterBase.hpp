#ifndef TENSOR_ADAPTER_BASE_HPP
#define TENSOR_ADAPTER_BASE_HPP

#include <vector>
#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/shape/Shape.hpp"
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
    virtual void addInPlace(const Tensor& other) = 0;
    virtual void subtractInPlace(const Tensor& other) = 0;
    virtual void multiplyInPlace(const Tensor& other) = 0;
    virtual void divideInPlace(const Tensor& other) = 0;

    // Comparison operator
    virtual bool equal(const Tensor& other) const = 0;
    virtual bool greaterThan(const Tensor& other) const = 0;
    virtual bool lessThan(const Tensor& other) const = 0;

    // Scalar operations
    virtual void addScalarInPlace(double scalar) = 0;
    virtual void subtractScalarInPlace(double scalar) = 0;
    virtual void multiplyScalarInPlace(double scalar) = 0;
    virtual void divideScalarInPlace(double scalar) = 0;

    // String representation.
    virtual std::string toString() = 0;
    virtual std::string toDataString() = 0;

    // Fill the tensor with a given value.
    virtual void fill(const DataItem& value) = 0;

    // Element access.
    virtual Tensor at(const std::vector<size_t>& indices) const = 0;
    virtual Tensor at(size_t index) const = 0;
    
    virtual void set(size_t index, const DataItem& value) = 0;
    virtual void set(const std::vector<size_t>& indices, const DataItem& value) = 0;

    virtual Tensor slice(const std::vector<std::pair<size_t, size_t>>& ranges) const = 0;

    // Shape operations.
    virtual void reshape(const Shape& newShape) = 0;

    // Memory management
    virtual std::unique_ptr<TensorAdapter> clone() const = 0;

    // Backend access
    virtual TensorBackend& backend() const = 0;

    virtual double getValueAsDouble(size_t index) const = 0;
    virtual void setValueFromDouble(size_t index, double value) = 0;

    virtual void getValueAsType(size_t index, const DataItem& data) const = 0;
    virtual void setValueFromType(size_t index, const DataItem& data) = 0;
};

} // namespace sdnn

#endif // TENSOR_ADAPTER_HPP
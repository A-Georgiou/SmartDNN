#ifndef CPU_TENSOR_BACKEND_HPP
#define CPU_TENSOR_BACKEND_HPP

#include "smart_dnn/shape/Shape.hpp"
#include "smart_dnn/tensor/TensorBackend.hpp" 
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <memory>
#include <vector>
#include <string>

namespace sdnn {

class Tensor;

class CPUTensorBackend : public TensorBackend {
public:
    CPUTensorBackend() = default;
    ~CPUTensorBackend();

    Tensor fill(const Shape& shape, const DataItem& value, dtype type) const override;

    template <typename T>
    Tensor fill(const Shape& shape, const T fillValue, dtype type) const{
        T fillValueTemp = fillValue;
        return this->fill(shape, {&fillValueTemp, dtype_trait<T>::value}, type);
    }

    // Basic operations
    Tensor add(const Tensor& a, const Tensor& b) const override;
    Tensor sub(const Tensor& a, const Tensor& b) const override;
    Tensor mul(const Tensor& a, const Tensor& b) const override;
    Tensor div(const Tensor& a, const Tensor& b) const override;

    Tensor add(const Tensor& a, const double& scalar) const override;
    Tensor sub(const Tensor& a, const double& scalar) const override;
    Tensor mul(const Tensor& a, const double& scalar) const override;
    Tensor div(const Tensor& a, const double& scalar) const override;

    Tensor scalarSub(const double& scalar, const Tensor& tensor) const override;
    Tensor scalarDiv(const double& scalar, const Tensor& tensor) const override;

    Tensor sumNoAxes(const Tensor& tensor) const;
    Tensor sum(const Tensor& tensor, const std::vector<int>& axes, bool keepDims) const override;
    Tensor meanNoAxes(const Tensor& tensor) const;
    Tensor mean(const Tensor& tensor, const std::vector<int>& axes, bool keepDims) const override;

    Tensor apply(const Tensor& tensor, const std::function<void(double&)>& func) const override;

    Tensor matmul(const Tensor& a, const Tensor& b) const override;

    Tensor reshape(const Tensor& tensor, const Shape& newShape) const override;
    Tensor transpose(const Tensor& tensor, const std::vector<int>& axes) const override;

    Tensor exp(const Tensor& tensor) const override;
    Tensor log(const Tensor& tensor) const override;
    Tensor power(const Tensor& tensor, double exponent) const override;
    Tensor sqrt(const Tensor& tensor) const override;
    Tensor abs(const Tensor& tensor) const override;
    Tensor negative(const Tensor& tensor) const override;

    bool equal(const Tensor& a, const Tensor& b) const override;
    bool greaterThan(const Tensor& a, const Tensor& b) const override;
    bool greaterThanEqual(const Tensor& a, const Tensor& b) const override;
    bool lessThan(const Tensor& a, const Tensor& b) const override;
    bool lessThanEqual(const Tensor& a, const Tensor& b) const override;

    Tensor rand(const Shape& shape, dtype type) const override;
    Tensor randn(const Shape& shape, dtype type, double min, double max) const override;

    Tensor zeros(const Shape& shape, dtype type) const override;
    Tensor zeros(int size, dtype type) const override;

    Tensor ones(const Shape& shape, dtype type) const override;
    Tensor ones(int size, dtype type) const override;

    Tensor identity(int size, dtype type) const override;
    
    // Backend information
    std::string backendName() const override;

    // Utility functions
    void print(const Tensor& tensor) override;
};

}; // namespace sdnn


#endif // CPU_TENSOR_BACKEND_HPP
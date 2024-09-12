
#include <memory>
#include "smart_dnn/tensor/Backend/Default/CPUTensorBackend.hpp"
#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/tensor/TensorBackend.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/tensor/Backend/Default/Utils.hpp"

namespace sdnn {

    CPUTensorBackend::~CPUTensorBackend() = default;


    Tensor CPUTensorBackend::createTensor(const Shape& shape, const void* data, dtype type) const {
        return Tensor(createCPUTensor(shape, static_cast<const sdnnTypeToPrimitive(type)>data));
    }

    Tensor CPUTensorBackend::fill(const Shape& shape, double value, dtype type) const {
        return Tensor(createTensorAdapter(shape, value, type));
    }

    Tensor CPUTensorBackend::add(const Tensor& a, const Tensor& b) const {
        Shape newShape = Shape({1});
        return Tensor(createTensorAdapter(newShape));
    }

    Tensor CPUTensorBackend::sub(const Tensor& a, const Tensor& b) const {
        Shape newShape = Shape({1});
        return Tensor(createTensorAdapter(newShape));
    }

    Tensor CPUTensorBackend::mul(const Tensor& a, const Tensor& b) const {
        Shape newShape = Shape({1});
        return Tensor(createTensorAdapter(newShape));
    }

    Tensor CPUTensorBackend::div(const Tensor& a, const Tensor& b) const {
        Shape newShape = Shape({1});
        return Tensor(createTensorAdapter(newShape));
    }

    Tensor CPUTensorBackend::add(const Tensor& a, const double& scalar) const {
        Shape newShape = Shape({1});
        return Tensor(createTensorAdapter(newShape));
    }

    Tensor CPUTensorBackend::sub(const Tensor& a, const double& scalar) const {
        Shape newShape = Shape({1});
        return Tensor(createTensorAdapter(newShape));
    }

    Tensor CPUTensorBackend::mul(const Tensor& a, const double& scalar) const {
        Shape newShape = Shape({1});
        return Tensor(createTensorAdapter(newShape));
    }

    Tensor CPUTensorBackend::div(const Tensor& a, const double& scalar) const {
        return Tensor(createTensorAdapter(Shape({1})));
    }

    Tensor CPUTensorBackend::scalarSub(const double& scalar, const Tensor& tensor) const {
       return Tensor(createTensorAdapter(Shape({1})));
    }

    Tensor CPUTensorBackend::scalarDiv(const double& scalar, const Tensor& tensor) const {
        return Tensor(createTensorAdapter(Shape({1})));
    }

    Tensor CPUTensorBackend::sum(const Tensor& tensor, const std::vector<int>& axes, bool keepDims) const {
        return Tensor(createTensorAdapter(tensor.shape()));
    }

    Tensor CPUTensorBackend::mean(const Tensor& tensor, const std::vector<int>& axes, bool keepDims) const {
        return Tensor(createTensorAdapter(tensor.shape()));
    }

    Tensor CPUTensorBackend::matmul(const Tensor& a, const Tensor& b) const {
        Shape newShape = Shape({1});
        return Tensor(createTensorAdapter(newShape));
    }

    Tensor CPUTensorBackend::reshape(const Tensor& tensor, const Shape& newShape) const {
        return Tensor(createTensorAdapter(tensor.shape()));
    }

    Tensor CPUTensorBackend::transpose(const Tensor& tensor, const std::vector<int>& axes) const {
        return Tensor(createTensorAdapter(tensor.shape()));
    }

    Tensor CPUTensorBackend::exp(const Tensor& tensor) const {
        return Tensor(createTensorAdapter(tensor.shape()));
    }

    Tensor CPUTensorBackend::log(const Tensor& tensor) const {
        return Tensor(createTensorAdapter(tensor.shape()));
    }

    Tensor CPUTensorBackend::power(const Tensor& tensor, double exponent) const {
        return Tensor(createTensorAdapter(tensor.shape()));
    }

    Tensor CPUTensorBackend::sqrt(const Tensor& tensor) const {
        return Tensor(createTensorAdapter(tensor.shape()));
    }

    Tensor CPUTensorBackend::abs(const Tensor& tensor) const {
        return Tensor(createTensorAdapter(tensor.shape()));
    }

    Tensor CPUTensorBackend::negative(const Tensor& tensor) const {
        return Tensor(createTensorAdapter(tensor.shape()));
    }

    bool CPUTensorBackend::equal(const Tensor& a, const Tensor& b) const {
        return a == b;
    }

    bool CPUTensorBackend::greaterThan(const Tensor& a, const Tensor& b) const {
        return true;
    }

    bool CPUTensorBackend::greaterThanEqual(const Tensor& a, const Tensor& b) const {
        return true;
    }

    bool CPUTensorBackend::lessThan(const Tensor& a, const Tensor& b) const {
        return true;
    }

    bool CPUTensorBackend::lessThanEqual(const Tensor& a, const Tensor& b) const {
        return true;
    }

    Tensor CPUTensorBackend::rand(const Shape& shape, dtype type) const {
        return Tensor(createTensorAdapter(shape));
    }

    Tensor CPUTensorBackend::randn(const Shape& shape, dtype type, double min, double max) const {
        return Tensor(createTensorAdapter(shape));
    }

    Tensor CPUTensorBackend::zeros(const Shape& shape, dtype type) const {
        return Tensor(createTensorAdapter(shape));
    }

    Tensor CPUTensorBackend::zeros(int size, dtype type) const {
        return Tensor(createTensorAdapter(Shape({size})));
    }

    Tensor CPUTensorBackend::ones(const Shape& shape, dtype type) const {
        return Tensor(createTensorAdapter(shape));
    }

    Tensor CPUTensorBackend::ones(int size, dtype type) const {
        return Tensor(createTensorAdapter(Shape({size})));
    }

    Tensor CPUTensorBackend::identity(int size, dtype type) const {
        return Tensor(createTensorAdapter(Shape({size, size})));
    }

    std::string CPUTensorBackend::backendName() const {
        return "CPUTensorBackend";
    }

    void CPUTensorBackend::print(const Tensor& tensor) {
        std::cout << tensor.toString() << std::endl;
    }

}; // namespace sdnn

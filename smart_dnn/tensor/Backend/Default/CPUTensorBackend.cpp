
#include <memory>
#include "smart_dnn/tensor/Backend/Default/CPUTensorBackend.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"
#include "smart_dnn/tensor/Backend/Default/TemplatedOperations.hpp"
#include "smart_dnn/DTypes.hpp"
#include "smart_dnn/tensor/TensorBackend.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/tensor/Backend/Default/Utils.hpp"
#include "smart_dnn/RandomEngine.hpp"
#include "smart_dnn/tensor/Backend/Default/AdvancedTensorOperations.hpp"
#include <vector>

namespace sdnn {

    CPUTensorBackend::~CPUTensorBackend() = default;

    Tensor CPUTensorBackend::fill(const Shape& shape, const double& value, dtype type) const {
        std::vector<double> data(shape.size(), value);
        auto tensorAdapter = createTensorAdapter(shape, data.data(), type);
        return Tensor(std::move(tensorAdapter));
    }

    Tensor CPUTensorBackend::add(const Tensor& a, const Tensor& b) const {
        return elementWiseOp(a, b, std::plus<>());
    }

    Tensor CPUTensorBackend::sub(const Tensor& a, const Tensor& b) const {
        return elementWiseOp(a, b, std::minus<>());
    }

    Tensor CPUTensorBackend::mul(const Tensor& a, const Tensor& b) const {
        return elementWiseOp(a, b, std::multiplies<>());
    }

    Tensor CPUTensorBackend::div(const Tensor& a, const Tensor& b) const {
        return elementWiseOp(a, b, std::divides<>());
    }

    Tensor CPUTensorBackend::add(const Tensor& a, const double& scalar) const {
        return scalarOp(a, scalar, std::plus<>());
    }

    Tensor CPUTensorBackend::sub(const Tensor& a, const double& scalar) const {
        return scalarOp(a, scalar, std::minus<>());
    }

    Tensor CPUTensorBackend::mul(const Tensor& a, const double& scalar) const {
        return scalarOp(a, scalar, std::multiplies<>());
    }

    Tensor CPUTensorBackend::div(const Tensor& a, const double& scalar) const {
        return scalarOp(a, scalar, std::divides<>());
    }

    Tensor CPUTensorBackend::scalarSub(const double& scalar, const Tensor& tensor) const {
        return scalarOp(tensor, scalar, [](auto a, auto b) { return b - a; });
    }

    Tensor CPUTensorBackend::scalarDiv(const double& scalar, const Tensor& tensor) const {
        return scalarOp(tensor, scalar, [](auto a, auto b) { return b / a; });
    }

    Tensor CPUTensorBackend::sum(const Tensor& tensor, const std::vector<int>& axes, bool keepDims) const {
        if (axes.empty()) {
            return sumNoAxes(tensor);
        }
        return reduction(tensor, axes, keepDims, 
                            [](auto a, auto b) { return a + b; },
                            [](auto sum, size_t) { return sum; });
    }

    Tensor CPUTensorBackend::sumNoAxes(const Tensor& tensor) const {
        double sum = 0.0;
        tensor.tensorImpl_->apply([&sum](double& a) { sum += a; });
        Shape newShape = Shape({1});
        return Tensor(createTensorAdapter(newShape, &sum, tensor.type()));
    }

    Tensor CPUTensorBackend::mean(const Tensor& tensor, const std::vector<int>& axes, bool keepDims) const {
        return reduction(tensor, axes, keepDims, 
                            [](auto a, auto b) { return a + b; },
                            [](auto sum, size_t count) { return sum / static_cast<decltype(sum)>(count); });
    }

    Tensor CPUTensorBackend::apply(const Tensor& tensor, const std::function<void(double&)>& func) const {
        auto output = tensor.tensorImpl_->clone();
        output->apply(func);
        return Tensor(std::move(output));
    }

    Tensor CPUTensorBackend::matmul(const Tensor& a, const Tensor& b) const {
        return AdvancedTensorOperations::matmul(a, b);
    }

    Tensor CPUTensorBackend::reshape(const Tensor& tensor, const Shape& newShape) const {
        auto output = tensor.tensorImpl_->clone();
        output->reshape(newShape);
        return Tensor(std::move(output));
    }

    Tensor CPUTensorBackend::transpose(const Tensor& tensor, const std::vector<int>& axes) const {
        const auto& shape = tensor.shape();
        const auto& type = tensor.type();
        
        if (axes.size() != shape.rank()) {
            throw std::invalid_argument("Number of axes must match tensor dimensions");
        }
        
        std::vector<int> newDims(shape.rank());
        std::vector<size_t> oldToNew(shape.rank());
        for (size_t i = 0; i < shape.rank(); ++i) {
            if (axes[i] < 0 || axes[i] >= shape.rank()) {
                throw std::invalid_argument("Invalid axis");
            }
            newDims[i] = shape[axes[i]];
            oldToNew[axes[i]] = i;
        }

        Shape newShape(newDims);
        auto output = createTensorAdapter(newShape, type);
        
        const size_t totalSize = shape.size();
        const auto& oldStrides = shape.getStride();
        const auto& newStrides = newShape.getStride();

        #pragma omp parallel for
        for (size_t i = 0; i < totalSize; ++i) {
            std::vector<size_t> oldIndices = unflattenIndex(i, shape);
            size_t newIndex = 0;
            for (size_t d = 0; d < shape.rank(); ++d) {
                newIndex += oldIndices[axes[d]] * newStrides[d];
            }
            
            double value = tensor.tensorImpl_->getValueAsDouble(i);
            output->setValueFromDouble(newIndex, value);
        }
        
        return Tensor(std::move(output));
    }

    Tensor CPUTensorBackend::exp(const Tensor& tensor) const {
        return applyOperation(tensor, [](auto a) { return std::exp(a); });
    }

    Tensor CPUTensorBackend::log(const Tensor& tensor) const {
        return applyOperation(tensor, [](auto a) { return std::log(a); });
    }

    Tensor CPUTensorBackend::power(const Tensor& tensor, double exponent) const {
        return applyOperation(tensor, [exponent](auto a) { return std::pow(a, exponent); });
    }

    Tensor CPUTensorBackend::sqrt(const Tensor& tensor) const {
        return applyOperation(tensor, [](auto a) { return std::sqrt(a); });
    }

    Tensor CPUTensorBackend::abs(const Tensor& tensor) const {
        return applyOperation(tensor, [](auto a) { 
            using T = decltype(a);

        if constexpr (std::is_unsigned_v<T> || std::is_same_v<T, bool>) {
            // For unsigned types and bool, return the value itself
            return a;
        } else {
            // For signed types, compute absolute value manually
            return a < T(0) ? -a : a;
        }
        });
    }

    Tensor CPUTensorBackend::negative(const Tensor& tensor) const {
        return applyOperation(tensor, [](auto a) { return -a; });
    }

    bool CPUTensorBackend::equal(const Tensor& a, const Tensor& b) const {
        return a == b;
    }

    bool CPUTensorBackend::greaterThan(const Tensor& a, const Tensor& b) const {
        return a.tensorImpl_->greaterThan(b);
    }

    bool CPUTensorBackend::greaterThanEqual(const Tensor& a, const Tensor& b) const {
        return a.tensorImpl_->greaterThan(b) || a.tensorImpl_->equal(b);
    }

    bool CPUTensorBackend::lessThan(const Tensor& a, const Tensor& b) const {
        return a.tensorImpl_->lessThan(b);
    }

    bool CPUTensorBackend::lessThanEqual(const Tensor& a, const Tensor& b) const {
        return a.tensorImpl_->lessThan(b) || a.tensorImpl_->equal(b);
    }

    Tensor CPUTensorBackend::rand(const Shape& shape, dtype type) const {
        std::vector<double> data(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            data[i] = static_cast<double>(RandomEngine::getRand());
        }
        return Tensor(createTensorAdapter(shape, data, type));
    }

    Tensor CPUTensorBackend::randn(const Shape& shape, dtype type, double min, double max) const {
        std::vector<double> data(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            data[i] = static_cast<double>(RandomEngine::getRandRange(min, max)) / RAND_MAX;
        }
        return Tensor(createTensorAdapter(shape, data, type));
    }

    Tensor CPUTensorBackend::zeros(const Shape& shape, dtype type) const {
        return fill(shape, 0.0, type);
    }

    Tensor CPUTensorBackend::zeros(int size, dtype type) const {
        return fill(Shape({size}), 0.0, type);
    }

    Tensor CPUTensorBackend::ones(const Shape& shape, dtype type) const {
        return fill(shape, 1.0, type);
    }

    Tensor CPUTensorBackend::ones(int size, dtype type) const {
        return fill(Shape({size}), 1.0, type);
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

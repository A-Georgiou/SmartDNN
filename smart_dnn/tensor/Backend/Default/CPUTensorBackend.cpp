
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

    #define IMPLEMENT_TYPE_SPECIFIC_OPS(TYPE) \
        Tensor CPUTensorBackend::add(const Tensor& a, const TYPE& scalar) const { \
            return scalarOp(a, scalar, std::plus<>());  \
        } \
        Tensor CPUTensorBackend::sub(const Tensor& a, const TYPE& scalar) const { \
            return scalarOp(a, scalar, std::minus<>()); \
        } \
        Tensor CPUTensorBackend::mul(const Tensor& a, const TYPE& scalar) const { \
            return scalarOp(a, scalar, std::multiplies<>());    \
        } \
        Tensor CPUTensorBackend::div(const Tensor& a, const TYPE& scalar) const { \
            return scalarOp(a, scalar, std::divides<>());   \
        } \
        Tensor CPUTensorBackend::scalarSub(const TYPE& scalar, const Tensor& a) const { \
            return scalarOp(a, scalar, [](auto a, auto b) { return b - a; });  \
        } \
        Tensor CPUTensorBackend::scalarDiv(const TYPE& scalar, const Tensor& a) const { \
            return scalarOp(a, scalar, [](auto a, auto b) { return b / a; });  \
        }  \
        Tensor CPUTensorBackend::fill(const Shape& shape, const TYPE& fillValue, dtype type) const { \
            auto tensorAdapter = createTensorAdapter(shape, type); \
            tensorAdapter->fill(fillValue); \
            return Tensor(std::move(tensorAdapter)); \
        } \

    // Generate scalar operations for various types
    IMPLEMENT_TYPE_SPECIFIC_OPS(bool)
    IMPLEMENT_TYPE_SPECIFIC_OPS(char)
    IMPLEMENT_TYPE_SPECIFIC_OPS(signed char)   // Maps to int8_t
    IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned char)
    IMPLEMENT_TYPE_SPECIFIC_OPS(short)         // 16-bit integer
    IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned short) // 16-bit unsigned integer
    IMPLEMENT_TYPE_SPECIFIC_OPS(int)
    IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned int)
    IMPLEMENT_TYPE_SPECIFIC_OPS(long)          // 64-bit integer on most systems
    IMPLEMENT_TYPE_SPECIFIC_OPS(unsigned long) // 64-bit unsigned integer on most systems
    IMPLEMENT_TYPE_SPECIFIC_OPS(float)
    IMPLEMENT_TYPE_SPECIFIC_OPS(double)

    #undef IMPLEMENT_TYPE_SPECIFIC_OPS


    Tensor CPUTensorBackend::sumNoAxes(const Tensor& tensor) const {
        double sum = 0.0f;
        CPUTensor tensorImpl = tensor.getImpl<CPUTensor>();
        tensorImpl.apply([&sum](auto& a) { sum += a; });
        return Tensor({1}, sum);
    }

    Tensor CPUTensorBackend::sum(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
        if (axes.empty()) {
            return sumNoAxes(tensor);
        }

        // Normalize axes
        std::vector<size_t> normalizedAxes;
        for (size_t axis : axes) {
            if (axis < 0) axis += tensor.shape().rank();
            if (axis < 0 || axis >= tensor.shape().rank()) {
                throw std::invalid_argument("Invalid axis in sum: " + std::to_string(axis));
            }
            normalizedAxes.push_back(axis);
        }


        float defaultValue = 0.0f;
        DataItem initialValue{&defaultValue, dtype::f32};

        auto result = reduction(tensor, normalizedAxes, keepDims, 
                                [](auto a, auto b) { return a + b; },
                                [](auto sum, size_t) { return sum; },
                                initialValue);

        return result;
    }

    Tensor CPUTensorBackend::meanNoAxes(const Tensor& tensor) const {
        Tensor sum = sumNoAxes(tensor);
        return sum / tensor.shape().size();
    }

    Tensor CPUTensorBackend::mean(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
        if (axes.empty()) {
            return meanNoAxes(tensor);
        }

        float defaultValue = 0;
        DataItem initialValue{&defaultValue, dtype::f32};

        return reduction(tensor, axes, keepDims, 
                            [](auto a, auto b) { return a + b; },
                            [](auto sum, size_t count) { return sum / static_cast<decltype(sum)>(count);},
                            initialValue);
    }

    Tensor CPUTensorBackend::max(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
        if (axes.empty()) {
            return maxNoAxes(tensor);
        }

        float lowestValue = std::numeric_limits<float>::lowest();
        DataItem initialValue{&lowestValue, dtype::f32};

        return reduction(tensor, axes, keepDims, 
                        [](auto a, auto b) { return std::max(a, b); },
                        [](auto max, size_t) { return max; },
                        initialValue);
    }

    Tensor CPUTensorBackend::selectMax(const Tensor& tensor, const double& min_value) const {
        return applyOperation(tensor, [min_value](auto a) { return a > min_value ? a : min_value; });
    }

    Tensor CPUTensorBackend::selectMax(const Tensor& a, const Tensor& b) const {
        return elementWiseOp(a, b, [](auto a, auto b) { return std::max(a, b); });
    }

    Tensor CPUTensorBackend::min(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
        if (axes.empty()) {
            return minNoAxes(tensor);
        }

        float maxValue = std::numeric_limits<float>::max();
        DataItem initialValue{&maxValue, dtype::f32};

        return reduction(tensor, axes, keepDims, 
                        [](auto a, auto b) { return std::min(a, b); },
                        [](auto min, size_t) { return min; },
                        initialValue);
    }

    Tensor CPUTensorBackend::maxNoAxes(const Tensor& tensor) const {
        auto maxTensor = tensor.getImpl<CPUTensor>();
        float maxVal = std::numeric_limits<float>::lowest();
        maxTensor.apply([&maxVal](auto a) { 
            maxVal = std::max(maxVal, static_cast<float>(a)); 
        });
        return Tensor({1}, maxVal);
    }

    Tensor CPUTensorBackend::minNoAxes(const Tensor& tensor) const {
        auto minTensor = tensor.getImpl<CPUTensor>();
        float minVal = std::numeric_limits<float>::max();
        minTensor.apply([&minVal](auto a) { 
            minVal = std::min(minVal, static_cast<float>(a)); 
        });
        return Tensor({1}, minVal);
    }

     Tensor CPUTensorBackend::clip(const Tensor& tensor, const double& min, const double& max) const {
        return applyOperation(tensor, [min, max](auto a) {
            using T = decltype(a);
            const T typedMin = static_cast<T>(min);
            const T typedMax = static_cast<T>(max);
            
            if (a < typedMin) return typedMin;
            if (a > typedMax) return typedMax;
            return a;
        });
        }

    Tensor CPUTensorBackend::apply(const Tensor& tensor, const std::function<void(double&)>& func) const {
        Tensor output = tensor.clone();
        auto outputImpl = tensor.getImpl<CPUTensor>();
        outputImpl.apply(func);
        return output;
    }

    Tensor CPUTensorBackend::select(const Tensor& condition, const Tensor& a, const Tensor& b) const {
        if (condition.shape() != a.shape() || condition.shape() != b.shape()) {
            throw std::invalid_argument("Select Error - All tensors must have the same shape, mismatch: " + 
                                        condition.shape().toString() + " != " + a.shape().toString() + " != " + b.shape().toString());
        }

        return selectOperation(condition, a, b, [](auto cond, auto a, auto b) {
            return cond == 1 ? a : b;
        });
    }

    Tensor CPUTensorBackend::matmul(const Tensor& a, const Tensor& b) const {
        return AdvancedTensorOperations::matmul(a, b);
    }

    Tensor CPUTensorBackend::reshape(const Tensor& tensor, const Shape& newShape) const {
        auto output = tensor.tensorImpl_->clone();
        output->reshape(newShape);
        return Tensor(std::move(output));
    }

    Tensor CPUTensorBackend::transpose(const Tensor& tensor, const std::vector<size_t>& axes) const {
        const auto& shape = tensor.shape();
        const auto& type = tensor.type();
        
        if (axes.size() != shape.rank()) {
            throw std::invalid_argument("Transpose Error - Number of axes must match tensor dimensions, mismatch: " + 
                                        std::to_string(axes.size()) + " != " + std::to_string(shape.rank()));
        }
        
        std::vector<int> newDims(shape.rank());
        for (size_t i = 0; i < shape.rank(); ++i) {
            newDims[i] = shape[axes[i]];
        }
        
        Shape newShape(newDims);
        Tensor output = zeros(newShape, type);
        
        const size_t totalSize = shape.size();
        const auto& oldStrides = shape.getStride();
        const auto& newStrides = newShape.getStride();

        #pragma omp parallel for
        for (size_t i = 0; i < totalSize; ++i) {
            size_t oldIndex = i;
            size_t newIndex = 0;

            for (size_t d = 0; d < shape.rank(); ++d) {
                size_t oldStride = oldIndex / oldStrides[d];
                oldIndex = oldIndex % oldStrides[d];
                newIndex += oldStride * newStrides[axes[d]];
            }
            
            double value = tensor.at<double>(i);
            output.set(newIndex, value);
        }
        
        return output;
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
                return a;
            } else {
                return a < T(0) ? -a : a;
            }
        });
    }

    Tensor CPUTensorBackend::tanh(const Tensor& tensor) const {
        return applyOperation(tensor, [](auto a) { return std::tanh(a); });
    }

    Tensor CPUTensorBackend::negative(const Tensor& tensor) const {
        return applyOperation(tensor, [](auto a) { return -a; });
    }

    Tensor CPUTensorBackend::variance(const Tensor& tensor, const Tensor& meanTensor, const std::vector<size_t>& axes) const {
        return AdvancedTensorOperations::variance(tensor, meanTensor, axes);
    }

    Tensor CPUTensorBackend::reciprocal(const Tensor& tensor, double epsilon) const {
        return AdvancedTensorOperations::reciprocal(tensor, epsilon);
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

    Tensor CPUTensorBackend::prodGreaterThan(const Tensor& a, const Tensor& b) const {
        return elementWiseOp(a, b, [](auto a, auto b) { return a > b; });
    }

    Tensor CPUTensorBackend::prodLessThan(const Tensor& a, const Tensor& b) const {
        return elementWiseOp(a, b, [](auto a, auto b) { return a < b; });
    }

    Tensor CPUTensorBackend::prodGreaterThan(const Tensor& a, const double& scalar) const {
        return applyOperation(a, [scalar](auto a) { return a > scalar; });
    }

    Tensor CPUTensorBackend::prodLessThan(const Tensor& a, const double& scalar) const {
        return applyOperation(a, [scalar](auto a) { return a < scalar; });
    }

    Tensor CPUTensorBackend::prodGreaterThanOrEqual(const Tensor& a, const double& scalar) const {
        return applyOperation(a, [scalar](auto x) { return x >= scalar; });
    }

    Tensor CPUTensorBackend::prodLessThanOrEqual(const Tensor& a, const double& scalar) const {
        return applyOperation(a, [scalar](auto x) { return x <= scalar; });
    }

    Tensor CPUTensorBackend::prodGreaterThanOrEqual(const Tensor& a, const Tensor& b) const {
        return elementWiseOp(a, b, [](auto a, auto b) { return a >= b; });
    }

    Tensor CPUTensorBackend::prodLessThanOrEqual(const Tensor& a, const Tensor& b) const {
        return elementWiseOp(a, b, [](auto a, auto b) { return a <= b; });
    }

    Tensor CPUTensorBackend::rand(const Shape& shape, dtype type) const {
        std::vector<float> data(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            data[i] = static_cast<float>(RandomEngine::getXavierInit(shape.size()));
        }
        return Tensor(createTensorAdapter(shape, data, type));
    }

    Tensor CPUTensorBackend::uniformRand(const Shape& shape, dtype type) const {
        std::vector<float> data(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            data[i] = static_cast<float>(RandomEngine::getRandRange(0.0f, 1.0f));
        }
        return Tensor(createTensorAdapter(shape, data, type));
    }

    Tensor CPUTensorBackend::randn(const Shape& shape, dtype type, float min, float max) const {
        std::vector<float> data(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            data[i] = static_cast<float>(RandomEngine::getHeRandRange(shape.size(), min, max));  // Removed RAND_MAX division
        }
        return Tensor(createTensorAdapter(shape, data, type));
    }

    Tensor CPUTensorBackend::zeros(const Shape& shape, dtype type) const {
        return this->fill(shape, 0, type);
    }

    Tensor CPUTensorBackend::zeros(int size, dtype type) const {
        return this->fill(Shape({size}), 0, type);
    }

    Tensor CPUTensorBackend::ones(const Shape& shape, dtype type) const {
        return this->fill(shape, 1, type);
    }

    Tensor CPUTensorBackend::ones(int size, dtype type) const {
        return this->fill(Shape({size}), 1, type);
    }

    Tensor CPUTensorBackend::identity(int size, dtype type) const {
        return Tensor(createTensorAdapter(Shape({size, size}), type));
    }

    std::string CPUTensorBackend::backendName() const {
        return "CPUTensorBackend";
    }

    void CPUTensorBackend::print(const Tensor& tensor) {
        std::cout << tensor.toString() << std::endl;
    }

}; // namespace sdnn

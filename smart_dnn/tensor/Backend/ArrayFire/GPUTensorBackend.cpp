#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.hpp"
#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensor.hpp"
#include "smart_dnn/tensor/Backend/ArrayFire/Utils.hpp"
#include <algorithm>

namespace sdnn {

   GPUTensorBackend::~GPUTensorBackend() = default;

    Tensor GPUTensorBackend::add(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        af::array result = a_cpu.getArray() + b_cpu.getArray();
        Shape resultShape = (a.shape() == b.shape()) ? a.shape() : ShapeOperations::broadcastShapes(a.shape(), b.shape());
        return Tensor(std::make_unique<GPUTensor>(resultShape, result, a.type()));
    }

    Tensor GPUTensorBackend::sub(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        af::array result = a_cpu.getArray() - b_cpu.getArray();
        Shape resultShape = (a.shape() == b.shape()) ? a.shape() : ShapeOperations::broadcastShapes(a.shape(), b.shape());
        return Tensor(std::make_unique<GPUTensor>(resultShape, result, a.type()));
    }

    Tensor GPUTensorBackend::mul(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        af::array result = a_cpu.getArray() * b_cpu.getArray();
        Shape resultShape = (a.shape() == b.shape()) ? a.shape() : ShapeOperations::broadcastShapes(a.shape(), b.shape());
        return Tensor(std::make_unique<GPUTensor>(resultShape, result, a.type()));
    }

    Tensor GPUTensorBackend::div(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        af::array result = a_cpu.getArray() / b_cpu.getArray();
        Shape resultShape = (a.shape() == b.shape()) ? a.shape() : ShapeOperations::broadcastShapes(a.shape(), b.shape());
        return Tensor(std::make_unique<GPUTensor>(resultShape, result, a.type()));
    }

    #define IMPLEMENT_TYPE_SPECIFIC_OPS(TYPE) \
        Tensor GPUTensorBackend::add(const Tensor& a, const TYPE& scalar) const { \
            GPUTensor tensor_cpu = a.getImpl<GPUTensor>();  \
            af::array result = tensor_cpu.getArray() + scalar;  \
            return Tensor(std::make_unique<GPUTensor>(a.shape(), result, a.type()));    \
        } \
        Tensor GPUTensorBackend::sub(const Tensor& a, const TYPE& scalar) const { \
            GPUTensor tensor_cpu = a.getImpl<GPUTensor>();  \
            af::array result = tensor_cpu.getArray() - scalar;  \
            return Tensor(std::make_unique<GPUTensor>(a.shape(), result, a.type()));    \
        } \
        Tensor GPUTensorBackend::mul(const Tensor& a, const TYPE& scalar) const { \
            GPUTensor tensor_cpu = a.getImpl<GPUTensor>();  \
            af::array result = tensor_cpu.getArray() * scalar;  \
            return Tensor(std::make_unique<GPUTensor>(a.shape(), result, a.type()));    \
        } \
        Tensor GPUTensorBackend::div(const Tensor& a, const TYPE& scalar) const { \
            GPUTensor tensor_cpu = a.getImpl<GPUTensor>();  \
            af::array result = tensor_cpu.getArray() / scalar;  \
            return Tensor(std::make_unique<GPUTensor>(a.shape(), result, a.type()));    \
        } \
        Tensor GPUTensorBackend::scalarSub(const TYPE& scalar, const Tensor& a) const { \
            GPUTensor tensor_cpu = a.getImpl<GPUTensor>();  \
            af::array result =  scalar - tensor_cpu.getArray();  \
            return Tensor(std::make_unique<GPUTensor>(a.shape(), result, a.type()));    \
        } \
        Tensor GPUTensorBackend::scalarDiv(const TYPE& scalar, const Tensor& a) const { \
            GPUTensor tensor_cpu = a.getImpl<GPUTensor>();  \
            af::array result =  scalar / tensor_cpu.getArray();  \
            return Tensor(std::make_unique<GPUTensor>(a.shape(), result, a.type()));    \
        }  \
        Tensor GPUTensorBackend::fill(const Shape& shape, const TYPE& fillValue, dtype type) const { \
            af::dtype afType = utils::sdnnToAfType(type); \
            std::vector<int> dimensions = shape.getDimensions(); \
            af::dim4 dims = utils::shapeToAfDim(shape); \
            af::array result = af::constant(fillValue, dims, afType); \
            return Tensor(std::make_unique<GPUTensor>(shape, result, type)); \
        } \

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

    Tensor GPUTensorBackend::sum(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        af::array result = tensor_cpu.getArray();
        
        // Sort axes in descending order to avoid weird index shifting (bloody ArrayFire)
        std::vector<size_t> sortedAxes = axes;
        std::sort(sortedAxes.rbegin(), sortedAxes.rend());
        
        for (size_t axis : sortedAxes) {
            result = af::sum(result, axis);
        }

        Shape resultShape;
        if (keepDims) {
            std::vector<int> newDims = tensor.shape().getDimensions();
            for (size_t axis : axes) {
                if (axis < newDims.size()) {
                    newDims[axis] = 1;
                }
            }
            resultShape = Shape(newDims);
            
            af::dim4 afDims = utils::shapeToAfDim(resultShape);
            result = af::moddims(result, afDims);
        } else {
            std::vector<int> newDims;
            const auto& inputDims = tensor.shape().getDimensions();
            for (size_t i = 0; i < inputDims.size(); ++i) {
                if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                    newDims.push_back(inputDims[i]);
                }
            }
            if (newDims.empty()) newDims.push_back(1);
            resultShape = Shape(newDims);
            
            af::dim4 afDims = utils::shapeToAfDim(resultShape);
            result = af::moddims(result, afDims);
        }

        return Tensor(std::make_unique<GPUTensor>(resultShape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::mean(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        af::array result = tensor_cpu.getArray();

        std::vector<size_t> sortedAxes = axes;
        std::sort(sortedAxes.rbegin(), sortedAxes.rend());
        
        for (size_t axis : sortedAxes) {
            result = af::mean(result, axis);
        }

        Shape resultShape;
        if (keepDims) {
            std::vector<int> newDims = tensor.shape().getDimensions();
            for (size_t axis : axes) {
                if (axis < newDims.size()) {
                    newDims[axis] = 1;
                }
            }
            resultShape = Shape(newDims);
        } else {
            std::vector<int> newDims;
            const auto& inputDims = tensor.shape().getDimensions();
            for (size_t i = 0; i < inputDims.size(); ++i) {
                if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                    newDims.push_back(inputDims[i]);
                }
            }
            if (newDims.empty()) newDims.push_back(1);
            resultShape = Shape(newDims);
        }

        return Tensor(std::make_unique<GPUTensor>(resultShape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::max(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        af::array result = tensor_cpu.getArray();

        std::vector<size_t> sortedAxes = axes;
        std::sort(sortedAxes.rbegin(), sortedAxes.rend());
        
        for (size_t axis : sortedAxes) {
            result = af::max(result, static_cast<int>(axis));
        }

        Shape resultShape;
        if (keepDims) {
            std::vector<int> newDims = tensor.shape().getDimensions();
            for (size_t axis : axes) {
                if (axis < newDims.size()) {
                    newDims[axis] = 1;
                }
            }
            resultShape = Shape(newDims);
        } else {
            std::vector<int> newDims;
            const auto& inputDims = tensor.shape().getDimensions();
            for (size_t i = 0; i < inputDims.size(); ++i) {
                if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                    newDims.push_back(inputDims[i]);
                }
            }
            if (newDims.empty()) newDims.push_back(1);
            resultShape = Shape(newDims);
        }

        return Tensor(std::make_unique<GPUTensor>(resultShape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::selectMax(const Tensor& tensor, const double& value) const {
        GPUTensor tensor_gpu = tensor.getImpl<GPUTensor>();
        af::array result = tensor_gpu.getArray();
        af::array selected = af::select(result >= value, result, 0.0);
        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(selected));
        return Tensor(std::make_unique<GPUTensor>(shape, selected, tensor.type()));
    }

    Tensor GPUTensorBackend::selectMax(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = af::max(aArray, bArray);  
        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(comparisonResult));
        return Tensor(std::make_unique<GPUTensor>(shape, comparisonResult, a.type()));
    }

    Tensor GPUTensorBackend::min(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        af::array result = tensor_cpu.getArray();

        // Apply min reduction across each axis
        for (size_t axis : axes) {
            result = af::min(result, static_cast<int>(axis));
        }

        if (keepDims) {
            for (size_t axis : axes) {
                result = af::moddims(result, result.dims(0), result.dims(1), result.dims(2), result.dims(3));  // Modify dims if necessary
            }
        }

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::clip(const Tensor& tensor, const double& min, const double& max) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        af::array result = tensor_cpu.getArray();

        result = af::clamp(result, min, max);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::matmul(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        af::array result = af::matmul(a_cpu.getArray(), b_cpu.getArray());

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, a.type()));
    }

    Tensor GPUTensorBackend::reshape(const Tensor& tensor, const Shape& newShape) const {
        auto output = tensor.tensorImpl_->clone();
        output->reshape(newShape);
        return Tensor(std::move(output));
    }
    
    Tensor GPUTensorBackend::transpose(const Tensor& tensor, const std::vector<size_t>& axes) const {
        const auto& shape = tensor.shape();
        
        if (axes.size() != shape.rank()) {
            throw std::invalid_argument("Transpose Error - Number of axes must match tensor dimensions, mismatch: " +
                                        std::to_string(axes.size()) + " != " + std::to_string(shape.rank()));
        }

        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();
        
        // af::reorder requires 4 indices. For tensors with rank < 4, we need to pad with identity permutation
        // E.g., for rank 2 with axes {1, 0}, we call af::reorder(arr, 1, 0, 2, 3)
        std::vector<unsigned> reorderAxes = {0, 1, 2, 3};
        for (size_t i = 0; i < axes.size(); ++i) {
            reorderAxes[i] = static_cast<unsigned>(axes[i]);
        }

        af::array result = af::reorder(inputArray, 
                                    reorderAxes[0], 
                                    reorderAxes[1], 
                                    reorderAxes[2], 
                                    reorderAxes[3]);

        std::vector<int> newDims(shape.rank());
        for (size_t i = 0; i < axes.size(); ++i) {
            newDims[i] = static_cast<int>(shape[axes[i]]);
        }
        Shape output_shape = Shape(newDims);
        
        return Tensor(std::make_unique<GPUTensor>(output_shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::reciprocal(const Tensor& tensor, double epsilon) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = 1.0 / (inputArray + epsilon);

        return Tensor(std::make_unique<GPUTensor>(tensor.shape(), result, tensor.type()));
    }

    Tensor GPUTensorBackend::exp(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::exp(inputArray);

        return Tensor(std::make_unique<GPUTensor>(tensor.shape(), result, tensor.type()));
    }

    Tensor GPUTensorBackend::log(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::log(inputArray);

        return Tensor(std::make_unique<GPUTensor>(tensor.shape(), result, tensor.type()));
    }

    Tensor GPUTensorBackend::power(const Tensor& tensor, double exponent) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::pow(inputArray, exponent);

        return Tensor(std::make_unique<GPUTensor>(tensor.shape(), result, tensor.type()));
    }

    Tensor GPUTensorBackend::sqrt(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::sqrt(inputArray);

        return Tensor(std::make_unique<GPUTensor>(tensor.shape(), result, tensor.type()));
    }

    Tensor GPUTensorBackend::abs(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::abs(inputArray);

        return Tensor(std::make_unique<GPUTensor>(tensor.shape(), result, tensor.type()));
    }

    Tensor GPUTensorBackend::tanh(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::tanh(inputArray);

        return Tensor(std::make_unique<GPUTensor>(tensor.shape(), result, tensor.type()));
    }

    Tensor GPUTensorBackend::negative(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = -inputArray;

        return Tensor(std::make_unique<GPUTensor>(tensor.shape(), result, tensor.type()));
    }

    Tensor GPUTensorBackend::variance(const Tensor& tensor, const Tensor& meanTensor, const std::vector<size_t>& axes) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        GPUTensor meanImpl = meanTensor.getImpl<GPUTensor>();

        af::array inputArray = tensorImpl.getArray();
        af::array meanArray = meanImpl.getArray();

        af::array diff = inputArray - meanArray;
        af::array squaredDiff = af::pow(diff, 2);

        af::array result = squaredDiff;

        std::vector<size_t> sortedAxes = axes;
        std::sort(sortedAxes.rbegin(), sortedAxes.rend());

        for (size_t axis : sortedAxes) {
            result = af::mean(result, static_cast<int>(axis));
        }

        std::vector<int> newDims;
        const auto& inputDims = tensor.shape().getDimensions();
        for (size_t i = 0; i < inputDims.size(); ++i) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                newDims.push_back(inputDims[i]);
            }
        }
        if (newDims.empty()) newDims.push_back(1);
        Shape resultShape(newDims);

        return Tensor(std::make_unique<GPUTensor>(resultShape, result, tensor.type()));
    }

    bool GPUTensorBackend::equal(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray == bArray);

        return af::allTrue<bool>(comparisonResult);
    }

    bool GPUTensorBackend::greaterThan(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        // Element-wise comparison for greater than
        af::array comparisonResult = (aArray > bArray);

        return af::allTrue<bool>(comparisonResult);
    }

    bool GPUTensorBackend::greaterThanEqual(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray >= bArray);

        return af::allTrue<bool>(comparisonResult);
    }

    bool GPUTensorBackend::lessThan(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray < bArray);

        return af::allTrue<bool>(comparisonResult);
    }

    bool GPUTensorBackend::lessThanEqual(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray <= bArray);

        return af::allTrue<bool>(comparisonResult);
    }

    Tensor GPUTensorBackend::select(const Tensor& condition, const Tensor& a, const Tensor& b) const {
        GPUTensor conditionImpl = condition.getImpl<GPUTensor>();
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();

        af::array conditionArray = conditionImpl.getArray();
        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array result = af::select(conditionArray, aArray, bArray);
        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, condition.type()));
    }

    Tensor GPUTensorBackend::prodGreaterThan(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();

        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray > bArray);
        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(comparisonResult));
        return Tensor(std::make_unique<GPUTensor>(shape, comparisonResult, a.type()));
    }

    Tensor GPUTensorBackend::prodLessThan(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();

        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray < bArray);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(comparisonResult));
        return Tensor(std::make_unique<GPUTensor>(shape, comparisonResult, a.type()));
    }

    Tensor GPUTensorBackend::prodGreaterThan(const Tensor& a, const double& scalar) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();

        af::array comparisonResult = (aArray > scalar);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(comparisonResult));
        return Tensor(std::make_unique<GPUTensor>(shape, comparisonResult, a.type()));
    }

    Tensor GPUTensorBackend::prodLessThan(const Tensor& a, const double& scalar) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();

        af::array comparisonResult = (aArray < scalar);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(comparisonResult));
        return Tensor(std::make_unique<GPUTensor>(shape, comparisonResult, a.type()));
    }

    Tensor GPUTensorBackend::prodGreaterThanOrEqual(const Tensor& a, const double& scalar) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();

        af::array comparisonResult = (aArray >= scalar);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(comparisonResult));
        return Tensor(std::make_unique<GPUTensor>(shape, comparisonResult, a.type()));
    }

    Tensor GPUTensorBackend::prodLessThanOrEqual(const Tensor& a, const double& scalar) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();

        af::array comparisonResult = (aArray <= scalar);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(comparisonResult));
        return Tensor(std::make_unique<GPUTensor>(shape, comparisonResult, a.type()));
    }

    Tensor GPUTensorBackend::prodGreaterThanOrEqual(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();

        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray >= bArray);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(comparisonResult));
        return Tensor(std::make_unique<GPUTensor>(shape, comparisonResult, a.type()));
    }

    Tensor GPUTensorBackend::prodLessThanOrEqual(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();

        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray <= bArray);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(comparisonResult));
        return Tensor(std::make_unique<GPUTensor>(shape, comparisonResult, a.type()));
    }

    Tensor GPUTensorBackend::rand(const Shape& shape, dtype type) const {
        std::vector<int> dimsVec = shape.getDimensions();

        af::dim4 dims(dimsVec.size() > 0 ? dimsVec[0] : 1,
                    dimsVec.size() > 1 ? dimsVec[1] : 1,
                    dimsVec.size() > 2 ? dimsVec[2] : 1,
                    dimsVec.size() > 3 ? dimsVec[3] : 1);
        
        af::dtype afType = utils::sdnnToAfType(type);
        af::array result = af::randu(dims, afType);

        Shape output_shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(output_shape, result, type));
    }

    Tensor GPUTensorBackend::uniformRand(const Shape& shape, dtype type) const {
        std::vector<int> dimsVec = shape.getDimensions();

        af::dim4 dims(dimsVec.size() > 0 ? dimsVec[0] : 1,
                    dimsVec.size() > 1 ? dimsVec[1] : 1,
                    dimsVec.size() > 2 ? dimsVec[2] : 1,
                    dimsVec.size() > 3 ? dimsVec[3] : 1);

        af::dtype afType = utils::sdnnToAfType(type);
        af::array result = af::randu(dims, afType);

        Shape output_shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(output_shape, result, type));
    }

    Tensor GPUTensorBackend::randn(const Shape& shape, dtype type, float min, float max) const {
        std::vector<int> dimsVec = shape.getDimensions();

        af::dim4 dims(dimsVec.size() > 0 ? dimsVec[0] : 1,
                    dimsVec.size() > 1 ? dimsVec[1] : 1,
                    dimsVec.size() > 2 ? dimsVec[2] : 1,
                    dimsVec.size() > 3 ? dimsVec[3] : 1);

        af::dtype afType = utils::sdnnToAfType(type);
        af::array result = af::randn(dims, afType);

        Shape output_shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(output_shape, result, type));
    }

    Tensor GPUTensorBackend::zeros(const Shape& shape, dtype type) const {
        std::vector<int> dimsVec = shape.getDimensions();

        af::dim4 dims(dimsVec.size() > 0 ? dimsVec[0] : 1,
                    dimsVec.size() > 1 ? dimsVec[1] : 1,
                    dimsVec.size() > 2 ? dimsVec[2] : 1,
                    dimsVec.size() > 3 ? dimsVec[3] : 1);

        af::dtype afType = utils::sdnnToAfType(type);
        af::array result = af::constant(0, dims, afType);
        return Tensor(std::make_unique<GPUTensor>(shape, result, type));
    }

    Tensor GPUTensorBackend::zeros(int size, dtype type) const {
        return zeros(Shape({size}), type);
    }

    Tensor GPUTensorBackend::ones(const Shape& shape, dtype type) const {
        std::vector<int> dimsVec = shape.getDimensions();

        af::dim4 dims(dimsVec.size() > 0 ? dimsVec[0] : 1,
                    dimsVec.size() > 1 ? dimsVec[1] : 1,
                    dimsVec.size() > 2 ? dimsVec[2] : 1,
                    dimsVec.size() > 3 ? dimsVec[3] : 1);

        af::dtype afType = utils::sdnnToAfType(type);
        af::array result = af::constant(1, dims, afType);

        Shape output_shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(output_shape, result, type));
    }

    Tensor GPUTensorBackend::ones(int size, dtype type) const {
        return ones(Shape({size}), type);
    }

    Tensor GPUTensorBackend::identity(int size, dtype type) const {
        af::dtype afType = utils::sdnnToAfType(type);
        af::array result = af::identity(size, afType);

        Shape output_shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(output_shape, result, type));
    }
    
    std::string GPUTensorBackend::backendName() const {
        return "GPUTensorBackend - ArrayFire";
    }

    void GPUTensorBackend::print(const Tensor& tensor) {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array array = tensorImpl.getArray();
        af::print("Tensor: ", array);
    }
}
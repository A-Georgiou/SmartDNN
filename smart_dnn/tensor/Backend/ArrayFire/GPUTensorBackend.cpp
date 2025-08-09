#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.hpp"
#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensor.hpp"
#include "smart_dnn/tensor/Backend/ArrayFire/Utils.hpp"

namespace sdnn {

   GPUTensorBackend::GPUTensorBackend() {
       // ArrayFire backend initialization is now handled in defaultTensorBackend()
   }

   GPUTensorBackend::~GPUTensorBackend() = default;

    Tensor GPUTensorBackend::add(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        af::array result = a_cpu.getArray() + b_cpu.getArray();
        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, a.type()));
    }

    Tensor GPUTensorBackend::sub(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        af::array result = a_cpu.getArray() - b_cpu.getArray();
        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, a.type()));
    }

    Tensor GPUTensorBackend::mul(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        af::array result = a_cpu.getArray() * b_cpu.getArray();
        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, a.type()));
    }

    Tensor GPUTensorBackend::div(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        af::array result = a_cpu.getArray() / b_cpu.getArray();
        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, a.type()));
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
        
        for (size_t axis : axes) {
            result = af::sum(result, axis);
        }

        if (keepDims) {
            for (size_t axis : axes) {
                result = af::moddims(result, result.dims(0), result.dims(1), result.dims(2), result.dims(3));  // Modify as needed
            }
        }

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::mean(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        af::array result = tensor_cpu.getArray();

        for (size_t axis : axes) {
            result = af::mean(result, axis);
        }

        if (keepDims) {
            for (size_t axis : axes) {
                result = af::moddims(result, result.dims(0), result.dims(1), result.dims(2), result.dims(3));  // Modify dims if necessary
            }
        }

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::max(const Tensor& tensor, const std::vector<size_t>& axes, bool keepDims) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        af::array result = tensor_cpu.getArray();

        for (size_t axis : axes) {
            result = af::max(result, static_cast<int>(axis));
        }

        if (keepDims) {
            for (size_t axis : axes) {
                result = af::moddims(result, result.dims(0), result.dims(1), result.dims(2), result.dims(3));  // Modify dims if necessary
            }
        }

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
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

        result = af::max(af::min(result, max), min);

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

        af::array result = af::reorder(inputArray, 
                                    static_cast<unsigned>(axes[0]), 
                                    static_cast<unsigned>(axes[1]), 
                                    static_cast<unsigned>(axes[2]), 
                                    static_cast<unsigned>(axes[3])); // For 4D or fewer arrays

        Shape output_shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(output_shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::reciprocal(const Tensor& tensor, double epsilon) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = 1.0 / (inputArray + epsilon);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::exp(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::exp(inputArray);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::log(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::log(inputArray);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::power(const Tensor& tensor, double exponent) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::pow(inputArray, exponent);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::sqrt(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::sqrt(inputArray);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::abs(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::abs(inputArray);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::tanh(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::tanh(inputArray);

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::negative(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = -inputArray;

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(result));
        return Tensor(std::make_unique<GPUTensor>(shape, result, tensor.type()));
    }

    Tensor GPUTensorBackend::variance(const Tensor& tensor, const Tensor& meanTensor, const std::vector<size_t>& axes) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        GPUTensor meanImpl = meanTensor.getImpl<GPUTensor>();

        af::array inputArray = tensorImpl.getArray();
        af::array meanArray = meanImpl.getArray();

        af::array diff = inputArray - meanArray;
        af::array squaredDiff = af::pow(diff, 2);

        af::array varianceArray = squaredDiff;

        for (size_t axis : axes) {
            varianceArray = af::mean(varianceArray, static_cast<int>(axis));
        }

        Shape shape = Shape(utils::getArrayDimensionsAsIntVector(varianceArray));
        return Tensor(std::make_unique<GPUTensor>(shape, varianceArray, tensor.type()));
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
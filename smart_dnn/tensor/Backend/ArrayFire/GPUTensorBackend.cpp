#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.hpp"
#include "smart_dnn/tensor/Backend/ArrayFire/GPUTensor.hpp"

namespace sdnn {

     Tensor GPUTensorBackend::fill(const Shape& shape, const DataItem& value, dtype type) const {
        auto tensorAdapter = createTensorAdapter(shape, type);
        tensorAdapter->fill(value);
        return Tensor(std::move(tensorAdapter));
    }

    Tensor GPUTensorBackend::add(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(a_cpu.getArray() + b_cpu.getArray()));
    }

    Tensor GPUTensorBackend::sub(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(a_cpu.getArray() - b_cpu.getArray()));
    }

    Tensor GPUTensorBackend::mul(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(a_cpu.getArray() * b_cpu.getArray()));
    }

    Tensor GPUTensorBackend::div(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(a_cpu.getArray() / b_cpu.getArray()));
    }

    Tensor GPUTensorBackend::add(const Tensor& a, const double& scalar) const {
        GPUTensor tensor_cpu = a.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(tensor_cpu.getArray() + scalar));
    }

    Tensor GPUTensorBackend::sub(const Tensor& a, const double& scalar) const {
        GPUTensor tensor_cpu = a.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(tensor_cpu.getArray() - scalar));
    }

    Tensor GPUTensorBackend::mul(const Tensor& a, const double& scalar) const {
        GPUTensor tensor_cpu = a.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(tensor_cpu.getArray() * scalar));
    }

    Tensor GPUTensorBackend::div(const Tensor& a, const double& scalar) const {
        GPUTensor tensor_cpu = a.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(tensor_cpu.getArray() / scalar));
    }

    Tensor GPUTensorBackend::scalarSub(const double& scalar, const Tensor& tensor) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(scalar - tensor_cpu.getArray()));
    }

    Tensor GPUTensorBackend::scalarDiv(const double& scalar, const Tensor& tensor) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(scalar / tensor_cpu.getArray()));
    }

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

        return Tensor(std::make_unique<GPUTensor>(result));
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

        return Tensor(std::make_unique<GPUTensor>(result));
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

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::selectMax(const Tensor& tensor, const double& value) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        af::array result = tensor_cpu.getArray();
        af::array mask = result >= value;
        return Tensor(std::make_unique<GPUTensor>(af::select(result, mask, af::array())));
    }

    Tensor GPUTensorBackend::selectMax(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        // Element-wise comparison for maximum
        af::array comparisonResult = af::max(aArray, bArray);  

        return Tensor(std::make_unique<GPUTensor>(comparisonResult));
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

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::clip(const Tensor& tensor, const double& min, const double& max) const {
        GPUTensor tensor_cpu = tensor.getImpl<GPUTensor>();
        af::array result = tensor_cpu.getArray();

        result = af::clamp(result, min, max);

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::matmul(const Tensor& a, const Tensor& b) const {
        GPUTensor a_cpu = a.getImpl<GPUTensor>();
        GPUTensor b_cpu = b.getImpl<GPUTensor>();
        return Tensor(std::make_unique<GPUTensor>(af::matmul(a_cpu.getArray(), b_cpu.getArray())));
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

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::reciprocal(const Tensor& tensor, double epsilon) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = 1.0 / (inputArray + epsilon);

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::exp(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::exp(inputArray);

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::log(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::log(inputArray);

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::power(const Tensor& tensor, double exponent) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::pow(inputArray, exponent);

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::sqrt(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::sqrt(inputArray);

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::abs(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::abs(inputArray);

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::tanh(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = af::tanh(inputArray);

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::negative(const Tensor& tensor) const {
        GPUTensor tensorImpl = tensor.getImpl<GPUTensor>();
        af::array inputArray = tensorImpl.getArray();

        af::array result = -inputArray;

        return Tensor(std::make_unique<GPUTensor>(result));
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

        return Tensor(std::make_unique<GPUTensor>(varianceArray));
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

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::prodGreaterThan(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();

        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray > bArray);
        return Tensor(std::make_unique<GPUTensor>(comparisonResult));
    }

    Tensor GPUTensorBackend::prodLessThan(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();

        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray < bArray);

        return Tensor(std::make_unique<GPUTensor>(comparisonResult));
    }

    Tensor GPUTensorBackend::prodGreaterThan(const Tensor& a, const double& scalar) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();

        af::array comparisonResult = (aArray > scalar);

        return Tensor(std::make_unique<GPUTensor>(comparisonResult));
    }

    Tensor GPUTensorBackend::prodLessThan(const Tensor& a, const double& scalar) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();

        af::array comparisonResult = (aArray < scalar);

        return Tensor(std::make_unique<GPUTensor>(comparisonResult));
    }

    Tensor GPUTensorBackend::prodGreaterThanOrEqual(const Tensor& a, const double& scalar) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();

        af::array comparisonResult = (aArray >= scalar);

        return Tensor(std::make_unique<GPUTensor>(comparisonResult));
    }

    Tensor GPUTensorBackend::prodLessThanOrEqual(const Tensor& a, const double& scalar) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        af::array aArray = aImpl.getArray();

        af::array comparisonResult = (aArray <= scalar);

        return Tensor(std::make_unique<GPUTensor>(comparisonResult));
    }

    Tensor GPUTensorBackend::prodGreaterThanOrEqual(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();

        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray >= bArray);

        return Tensor(std::make_unique<GPUTensor>(comparisonResult));
    }

    Tensor GPUTensorBackend::prodLessThanOrEqual(const Tensor& a, const Tensor& b) const {
        GPUTensor aImpl = a.getImpl<GPUTensor>();
        GPUTensor bImpl = b.getImpl<GPUTensor>();

        af::array aArray = aImpl.getArray();
        af::array bArray = bImpl.getArray();

        af::array comparisonResult = (aArray <= bArray);

        return Tensor(std::make_unique<GPUTensor>(comparisonResult));
    }

    Tensor GPUTensorBackend::rand(const Shape& shape, dtype type) const {
        std::vector<int> dimsVec = shape.getDimensions();

        af::dim4 dims(dimsVec.size() > 0 ? dimsVec[0] : 1,
                    dimsVec.size() > 1 ? dimsVec[1] : 1,
                    dimsVec.size() > 2 ? dimsVec[2] : 1,
                    dimsVec.size() > 3 ? dimsVec[3] : 1);

        af::array result = af::randu(dims, static_cast<af::dtype>(type));

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::uniformRand(const Shape& shape, dtype type) const {
        std::vector<int> dimsVec = shape.getDimensions();

        af::dim4 dims(dimsVec.size() > 0 ? dimsVec[0] : 1,
                    dimsVec.size() > 1 ? dimsVec[1] : 1,
                    dimsVec.size() > 2 ? dimsVec[2] : 1,
                    dimsVec.size() > 3 ? dimsVec[3] : 1);

        af::array result = af::randu(dims, static_cast<af::dtype>(type));

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::randn(const Shape& shape, dtype type, float min, float max) const {
        std::vector<int> dimsVec = shape.getDimensions();

        af::dim4 dims(dimsVec.size() > 0 ? dimsVec[0] : 1,
                    dimsVec.size() > 1 ? dimsVec[1] : 1,
                    dimsVec.size() > 2 ? dimsVec[2] : 1,
                    dimsVec.size() > 3 ? dimsVec[3] : 1);

        af::array result = af::randn(dims, static_cast<af::dtype>(type));

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::zeros(const Shape& shape, dtype type) const {
        std::vector<int> dimsVec = shape.getDimensions();

        af::dim4 dims(dimsVec.size() > 0 ? dimsVec[0] : 1,
                    dimsVec.size() > 1 ? dimsVec[1] : 1,
                    dimsVec.size() > 2 ? dimsVec[2] : 1,
                    dimsVec.size() > 3 ? dimsVec[3] : 1);

        af::array result = af::constant(0.0, dims, static_cast<af::dtype>(type));

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::ones(const Shape& shape, dtype type) const {
        std::vector<int> dimsVec = shape.getDimensions();

        af::dim4 dims(dimsVec.size() > 0 ? dimsVec[0] : 1,
                    dimsVec.size() > 1 ? dimsVec[1] : 1,
                    dimsVec.size() > 2 ? dimsVec[2] : 1,
                    dimsVec.size() > 3 ? dimsVec[3] : 1);

        af::array result = af::constant(1.0, dims, static_cast<af::dtype>(type));

        return Tensor(std::make_unique<GPUTensor>(result));
    }

    Tensor GPUTensorBackend::identity(int size, dtype type) const {
        af::array result = af::identity(size, static_cast<af::dtype>(type));

        return Tensor(std::make_unique<GPUTensor>(result));
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
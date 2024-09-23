
#include <vector>
#include <memory>
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"
#include "smart_dnn/tensor/Backend/Default/BroadcastView.hpp"

namespace sdnn {

template<typename Op>
Tensor applyOperation(const Tensor& a, Op operation) {
    auto result = std::make_unique<CPUTensor>(a.shape(), a.type());
    const auto& a_cpu = a.getImpl<CPUTensor>();

    result->applyTypedOperation([&](auto type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* a_data = a_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const size_t size = a.shape().size();

        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = operation(a_data[i]);
        }
    });

    return Tensor(std::move(result));
}

template<typename Op>
Tensor elementWiseOp(const Tensor& a, const Tensor& b, Op operation) {
    Shape broadcastShape = ShapeOperations::broadcastShapes(a.shape(), b.shape());
    auto result = std::make_unique<CPUTensor>(broadcastShape, a.type());
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        
        BroadcastView<T> viewA(a, broadcastShape);
        BroadcastView<T> viewB(b, broadcastShape);
        T* result_data = result->typedData<T>();
        const size_t size = broadcastShape.size();

        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = operation(viewA[i], viewB[i]);
        }
    });

    return Tensor(std::move(result));
}

template<typename Op>
Tensor scalarOp(const Tensor& a, const double& scalar, Op operation) {
    auto result = std::make_unique<CPUTensor>(a.shape(), a.type());
    const auto& a_cpu = a.getImpl<CPUTensor>();

    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* a_data = a_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const T scalar_t = static_cast<T>(scalar);
        const size_t size = a.shape().size();

        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = operation(a_data[i], scalar_t);
        }
    });

    return Tensor(std::move(result));
}

template<typename Op, typename Finalize>
Tensor reduction(const Tensor& tensor, const std::vector<int>& axes, bool keepDims, Op operation, Finalize finalize) {
    const auto& input = tensor.getImpl<CPUTensor>();
    const auto& inputShape = tensor.shape();
    const size_t inputRank = inputShape.rank();

    // Normalize axes (handle negative axes)
    std::vector<int> normalizedAxes = axes;
    for (auto& axis : normalizedAxes) {
        if (axis < 0) axis += inputRank;
        if (axis < 0 || axis >= inputRank) {
            throw std::invalid_argument("Invalid axis for reduction");
        }
    }

    // Sort and remove duplicates from axes
    std::sort(normalizedAxes.begin(), normalizedAxes.end());
    normalizedAxes.erase(std::unique(normalizedAxes.begin(), normalizedAxes.end()), normalizedAxes.end());

    // Create the output shape based on the reduced axes
    std::vector<int> outputShape;
    for (int i = 0; i < inputRank; ++i) {
        if (std::find(normalizedAxes.begin(), normalizedAxes.end(), i) == normalizedAxes.end()) {
            outputShape.push_back(inputShape[i]);
        } else if (keepDims) {
            outputShape.push_back(1);
        }
    }

    // Create the output tensor
    auto result = std::make_unique<CPUTensor>(Shape(outputShape), tensor.type());

    // Compute input strides
    std::vector<size_t> inputStrides(inputRank, 1);
    for (int i = inputRank - 2; i >= 0; --i) {
        inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1];
    }

    // Compute the total size of the output tensor
    const size_t outputSize = result->size();

    // Perform the reduction
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* inputData = input.typedData<T>();
        T* outputData = result->typedData<T>();

        // Iterate over the output tensor and accumulate values along the reduction axes
        #pragma omp parallel for
        for (size_t outputIndex = 0; outputIndex < outputSize; ++outputIndex) {
            T accumulator = 0; // Accumulator for reduction

            // Use index-mapping to calculate the corresponding indices in the input tensor
            std::vector<size_t> outputCoords(inputRank, 0);
            size_t currentOutputIndex = outputIndex;

            for (int i = outputShape.size() - 1; i >= 0; --i) {
                if (outputShape[i] > 1) {
                    outputCoords[i] = currentOutputIndex % outputShape[i];
                    currentOutputIndex /= outputShape[i];
                }
            }

            // Perform reduction over the axes
            size_t count = 0;
            std::vector<size_t> reduceCoords = outputCoords;
            size_t numReduceElements = 1;
            for (int axis : normalizedAxes) {
                numReduceElements *= inputShape[axis];
            }

            for (size_t r = 0; r < numReduceElements; ++r) {
                size_t inputIndex = 0;
                size_t reduceIndex = r;

                // Calculate the actual input index by modifying reduceCoords
                for (int axis : normalizedAxes) {
                    size_t reduceDimSize = inputShape[axis];
                    reduceCoords[axis] = reduceIndex % reduceDimSize;
                    reduceIndex /= reduceDimSize;
                }

                // Compute the flattened input index based on reduceCoords
                for (size_t i = 0; i < inputRank; ++i) {
                    inputIndex += reduceCoords[i] * inputStrides[i];
                }

                // Apply the operation (e.g., sum, max, etc.)
                if (count == 0) {
                    accumulator = inputData[inputIndex];
                } else {
                    accumulator = operation(accumulator, inputData[inputIndex]);
                }
                ++count;
            }

            // Finalize the value and store it in the output tensor
            outputData[outputIndex] = finalize(accumulator, count);
        }
    });

    return Tensor(std::move(result));
}

}
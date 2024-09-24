
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

        #pragma omp simd
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

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = operation(viewA[i], viewB[i]);
        }
    });

    return Tensor(std::move(result));
}

template<typename U, typename Op>
Tensor scalarOp(const Tensor& a, const U& scalar, Op operation) {
    auto result = std::make_unique<CPUTensor>(a.shape(), a.type());
    const auto& a_cpu = a.getImpl<CPUTensor>();

    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* a_data = a_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        const T scalar_t = static_cast<T>(scalar);
        const size_t size = a.shape().size();

        #pragma omp simd
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

    // Calculate the size of the axes being reduced (for finalization)
    size_t reductionSize = 1;
    for (const int axis : normalizedAxes) {
        reductionSize *= inputShape[axis];
    }

    // Create the output shape based on the reduced axes
    std::vector<int> outputShape;
    for (int i = 0; i < inputRank; ++i) {
        if (std::find(normalizedAxes.begin(), normalizedAxes.end(), i) == std::end(normalizedAxes)) {
            outputShape.push_back(inputShape[i]);
        } else if (keepDims) {
            outputShape.push_back(1);
        }
    }

    // Create the output tensor
    auto result = std::make_unique<CPUTensor>(Shape(outputShape), tensor.type());

    // Perform the reduction by iterating over the input tensor
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* inputData = input.typedData<T>();
        T* outputData = result->typedData<T>();

        std::vector<size_t> inputCoords(inputRank, 0);
        const size_t outputSize = result->size();
        const size_t inputSize = input.size();

        // Initialize output with the first value for accumulation
        std::fill(outputData, outputData + outputSize, T(0));

        for (size_t i = 0; i < inputSize; ++i) {
            size_t outputIndex = 0;
            size_t stride = 1;

            // Calculate the output index based on inputCoords excluding the reduction axes
            for (int j = inputRank - 1; j >= 0; --j) {
                if (std::find(normalizedAxes.begin(), normalizedAxes.end(), j) == std::end(normalizedAxes)) {
                    outputIndex += inputCoords[j] * stride;
                    stride *= inputShape[j];
                }
            }

            // Apply the operation to accumulate values
            outputData[outputIndex] = operation(outputData[outputIndex], inputData[i]);

            // Update inputCoords for next iteration
            for (int j = inputRank - 1; j >= 0; --j) {
                inputCoords[j]++;
                if (inputCoords[j] < inputShape[j]) {
                    break;
                }
                inputCoords[j] = 0;
            }
        }

        for (size_t i = 0; i < outputSize; ++i) {
            outputData[i] = finalize(outputData[i], reductionSize);  // Use reductionSize here
        }
    });

    return Tensor(std::move(result));
}

}
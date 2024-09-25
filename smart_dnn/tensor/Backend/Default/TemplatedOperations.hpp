
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
    
    dtype promotedType = promotionOfTypes(a.type(), b.type());
    auto result = std::make_unique<CPUTensor>(broadcastShape, promotedType);
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using PromotedT = std::remove_pointer_t<decltype(type_ptr)>;
        
        const size_t size = broadcastShape.size();
        PromotedT* result_data = result->typedData<PromotedT>();

        auto processElement = [&](size_t i, auto viewA, auto viewB) {
            result_data[i] = operation(static_cast<PromotedT>(viewA[i]), static_cast<PromotedT>(viewB[i]));
        };

        applyTypedOperationHelper(a.type(), [&](auto dummy_a) {
            using TypeA = decltype(dummy_a);
            BroadcastView<TypeA> viewA(a, broadcastShape);

            applyTypedOperationHelper(b.type(), [&](auto dummy_b) {
                using TypeB = decltype(dummy_b);
                BroadcastView<TypeB> viewB(b, broadcastShape);

                #pragma omp simd
                for (size_t i = 0; i < size; ++i) {
                    processElement(i, viewA, viewB);
                }
            });
        });
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
Tensor reduction(const Tensor& tensor, const std::vector<int>& axes, bool keepDims, 
                 Op operation, Finalize finalize, const DataItem& initialValue) {
    const auto& input = tensor.getImpl<CPUTensor>();
    const auto& inputShape = tensor.shape();
    const size_t inputRank = inputShape.rank();

    // Normalize axes
    std::vector<int> normalizedAxes = axes;
    for (auto& axis : normalizedAxes) {
        if (axis < 0) axis += inputRank;
        if (axis < 0 || axis >= inputRank) {
            throw std::invalid_argument("Invalid axis for reduction");
        }
    }

    // Sort and remove duplicates
    std::sort(normalizedAxes.begin(), normalizedAxes.end());
    normalizedAxes.erase(std::unique(normalizedAxes.begin(), normalizedAxes.end()), normalizedAxes.end());

    std::vector<int> outputShape;
    for (size_t i = 0; i < inputRank; ++i) {
        if (std::find(normalizedAxes.begin(), normalizedAxes.end(), i) == normalizedAxes.end()) {
            outputShape.push_back(inputShape[i]);
        } else if (keepDims) {
            outputShape.push_back(1);
        }
    }
    if (outputShape.empty()) outputShape.push_back(1);

    // Create the output tensor
    auto result = std::make_unique<CPUTensor>(Shape(outputShape), tensor.type());
    
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T* outputData = result->typedData<T>();
        const size_t outputSize = result->size();
        T typedInitialValue = dtype_cast<T>(initialValue.data, initialValue.type);
        std::fill(outputData, outputData + outputSize, typedInitialValue);
    });

    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* inputData = input.typedData<T>();
        T* outputData = result->typedData<T>();

        const size_t outputSize = result->size();
        const size_t inputSize = input.size();

        std::vector<size_t> inputCoords(inputRank, 0);

        for (size_t i = 0; i < inputSize; ++i) {
            size_t outputIndex = 0;
            size_t outputStride = 1;

            for (int j = inputRank - 1; j >= 0; --j) {
                if (std::find(normalizedAxes.begin(), normalizedAxes.end(), j) == normalizedAxes.end()) {
                    outputIndex += inputCoords[j] * outputStride;
                    outputStride *= outputShape[j];
                } else if (keepDims) {
                    outputStride *= 1; 
                }
            }

            outputData[outputIndex] = operation(outputData[outputIndex], inputData[i]);

            for (int j = inputRank - 1; j >= 0; --j) {
                if (++inputCoords[j] < inputShape[j]) break;
                inputCoords[j] = 0;
            }
        }
        
        size_t reductionSize = 1;
        for (int axis : normalizedAxes) reductionSize *= inputShape[axis];
        for (size_t i = 0; i < outputSize; ++i) {
            outputData[i] = finalize(outputData[i], reductionSize);
        }
    });

    return Tensor(std::move(result));
}

}
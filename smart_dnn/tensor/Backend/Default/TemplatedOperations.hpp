
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

    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* a_data = a_cpu.typedData<T>();
        T* result_data = result->typedData<T>();

        for (size_t i = 0; i < a.shape().size(); ++i) {
            // Add an explicit conversion for boolean types (dtype::b8).
            if constexpr (std::is_same_v<T, bool>) {
                result_data[i] = static_cast<bool>(operation(a_data[i]));
            } else {
                result_data[i] = operation(a_data[i]);
            }
        }
    });

    return Tensor(std::move(result));
}

template<typename Op>
Tensor elementWiseOp(const Tensor& a, const Tensor& b, Op operation) {
    const Shape& shapeA = a.shape();
    const Shape& shapeB = b.shape();
    Shape broadcastShape = ShapeOperations::broadcastShapes(shapeA, shapeB);

    auto result = std::make_unique<CPUTensor>(broadcastShape, a.type());

    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        
        BroadcastView<T> viewA(a, broadcastShape);
        BroadcastView<T> viewB(b, broadcastShape);

        T* result_data = result->typedData<T>();

        for (size_t i = 0; i < broadcastShape.size(); ++i) {
            result_data[i] = operation(viewA[i], viewB[i]);
        }
    });

    return Tensor(std::move(result));
}


template<typename Op>
void applyBroadcastView(const Tensor& tensor, const Shape& broadcastShape, Op op) {
    applyTypedOperationHelper(tensor.type(), [&](auto dummy) {
        using T = decltype(dummy);
        BroadcastView<T> view(tensor, broadcastShape);
        op(view);
    });
}

template<typename Op>
Tensor scalarOp(const Tensor& a, const double& scalar, Op operation) {
    auto result = std::make_unique<CPUTensor>(a.shape(), a.type());
    const auto& a_cpu = a.getImpl<CPUTensor>();

    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* a_data = a_cpu.typedData<T>();
        T* result_data = result->typedData<T>();
        T scalar_t = static_cast<T>(scalar);

        for (size_t i = 0; i < a.shape().size(); ++i) {
            result_data[i] = operation(a_data[i], scalar_t);
        }
    });

    return Tensor(std::move(result));
}

template<typename Op, typename Finalize>
Tensor reduction(const Tensor& tensor, const std::vector<int>& axes, bool keepDims, Op operation, Finalize finalize) {
    const auto& input = tensor.getImpl<CPUTensor>();
    const auto& inputShape = tensor.shape();
    const size_t inputRank = inputShape.size();

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

    // Compute output shape
    std::vector<int> outputShape;
    for(int i = 0; i < inputRank; ++i) {
        if (std::find(normalizedAxes.begin(), normalizedAxes.end(), i) == normalizedAxes.end()) {
            outputShape.push_back(inputShape[i]);
        } else if (keepDims) {
            outputShape.push_back(1);
        }
    }

    // Create output tensor
    auto result = std::make_unique<CPUTensor>(Shape(outputShape), tensor.type());

    // Compute strides for input and output
    std::vector<size_t> inputStrides(inputRank, 1);
    std::vector<size_t> outputStrides(outputShape.size(), 1);
    for (int i = inputRank - 2; i >= 0; --i) {
        inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1];
    }
    for (int i = outputShape.size() - 2; i >= 0; --i) {
        outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];
    }

    // Perform reduction
    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* inputData = input.typedData<T>();
        T* outputData = result->typedData<T>();

        size_t outputSize = result->size();
        std::vector<size_t> currentIndex(inputRank, 0);

        for (size_t i = 0; i < outputSize; ++i) {
            // Compute the range for this output element
            size_t start = 0;
            size_t end = 1;
            for (int axis : normalizedAxes) {
                start *= inputShape[axis];
                end *= inputShape[axis];
            }

            // Perform reduction for this output element
            T accumulator = inputData[start];
            size_t count = 1;
            for (size_t j = start + 1; j < end; ++j) {
                size_t inputIndex = 0;
                for (size_t k = 0; k < inputRank; ++k) {
                    if (std::find(normalizedAxes.begin(), normalizedAxes.end(), k) == normalizedAxes.end()) {
                        inputIndex += currentIndex[k] * inputStrides[k];
                    } else {
                        inputIndex += (j / inputStrides[k] % inputShape[k]) * inputStrides[k];
                    }
                }
                accumulator = operation(accumulator, inputData[inputIndex]);
                ++count;
            }

            // Finalize and store the result
            outputData[i] = finalize(accumulator, count);

            // Update current index
            for (int j = outputShape.size() - 1; j >= 0; --j) {
                if (++currentIndex[j] < outputShape[j]) {
                    break;
                }
                currentIndex[j] = 0;
            }
        }
    });

    return Tensor(std::move(result));
}

}
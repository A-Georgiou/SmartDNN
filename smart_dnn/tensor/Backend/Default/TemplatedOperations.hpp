
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

    std::vector<int> normalizedAxes = axes;
    for (auto& axis : normalizedAxes) {
        if (axis < 0) axis += inputRank;
        if (axis < 0 || axis >= inputRank) {
            throw std::invalid_argument("Invalid axis for reduction");
        }
    }

    std::sort(normalizedAxes.begin(), normalizedAxes.end());
    normalizedAxes.erase(std::unique(normalizedAxes.begin(), normalizedAxes.end()), normalizedAxes.end());

    std::vector<int> outputShape;
    for(int i = 0; i < inputRank; ++i) {
        if (std::find(normalizedAxes.begin(), normalizedAxes.end(), i) == normalizedAxes.end()) {
            outputShape.push_back(inputShape[i]);
        } else if (keepDims) {
            outputShape.push_back(1);
        }
    }

    auto result = std::make_unique<CPUTensor>(Shape(outputShape), tensor.type());

    std::vector<size_t> inputStrides(inputRank, 1);
    for (int i = inputRank - 2; i >= 0; --i) {
        inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1];
    }

    result->applyTypedOperation([&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* inputData = input.typedData<T>();
        T* outputData = result->typedData<T>();

        const size_t outputSize = result->size();
        std::vector<size_t> currentIndex(inputRank, 0);

        #pragma omp parallel for
        for (size_t i = 0; i < outputSize; ++i) {
            size_t start = 0;
            size_t end = 1;
            for (int axis : normalizedAxes) {
                start *= inputShape[axis];
                end *= inputShape[axis];
            }

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

            outputData[i] = finalize(accumulator, count);

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
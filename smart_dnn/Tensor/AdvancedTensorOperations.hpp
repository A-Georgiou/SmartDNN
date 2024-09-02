#ifndef ADVANCED_TENSOR_OPERATIONS_HPP
#define ADVANCED_TENSOR_OPERATIONS_HPP

#include "TensorData.hpp"
#include "DeviceTypes.hpp"
#include "Tensor.hpp"
#include "../Shape/ShapeOperations.hpp"

namespace smart_dnn {

// Forward declaration of the primary template
template <typename T, typename DeviceType>
class AdvancedTensorOperations;

// Specialization for CPUDevice
template <typename T>
class AdvancedTensorOperations<T, CPUDevice> {
    public:
    AdvancedTensorOperations() = delete;

    static Tensor<T, DeviceType> matmul(const Tensor<T, DeviceType>& a, const Tensor<T, DeviceType>& b) {
        const auto& shapeA = a.getShape();
        const auto& shapeB = b.getShape();

        if (shapeA.rank() == 1 && shapeB.rank() == 1) {
            return dotProduct(a, b);
        } else if (shapeA.rank() == 2 && shapeB.rank() == 2) {
            return matrixMatrixMul(a, b);
        } else if (shapeA.rank() == 1 && shapeB.rank() == 2) {
            auto a_reshaped = reshape(a, {1, shapeA[0]});
            auto result = matrixMatrixMul(a_reshaped, b);
            return reshape(result, {result.getShape()[1]});
        } else if (shapeA.rank() == 2 && shapeB.rank() == 1) {
            return matrixVectorMul(a, b);
        } else if (shapeA.rank() > 2 || shapeB.rank() > 2) {
            return batchedMatmul(a, b);
        }

        throw std::invalid_argument("Invalid tensor ranks for matrix multiplication.");
    }

    private:
    static constexpr int TILE_SIZE = 32; 

    static Tensor<T, DeviceType> dotProduct(const Tensor<T, DeviceType>& a, const Tensor<T, DeviceType>& b) {
        if (a.getShape()[0] != b.getShape()[0]) {
            throw std::invalid_argument("Vector dimensions must match for dot product.");
        }

        T result = 0;
        const T* aData = a.getData().data();
        const T* bData = b.getData().data();
        int size = a.getShape()[0];

        #pragma omp simd reduction(+:result)
        for (int i = 0; i < size; ++i) {
            result += aData[i] * bData[i];
        }

        return Tensor<T, DeviceType>({1}, {result});
    }

    static Tensor<T, DeviceType> matrixVectorMul(const Tensor<T, DeviceType>& a, const Tensor<T, DeviceType>& b) {
        if (a.getShape()[1] != b.getShape()[0]) {
            throw std::invalid_argument("Matrix and vector dimensions must match for multiplication.");
        }

        int m = a.getShape()[0];
        int n = a.getShape()[1];
        Tensor<T, DeviceType> result({m}, T(0));

        const T* aData = a.getData().data();
        const T* bData = b.getData().data();
        T* resultData = result.getData().data();

        #pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < n; ++j) {
                sum += aData[i * n + j] * bData[j];
            }
            resultData[i] = sum;
        }

        return result;
    }

    static Tensor<T, DeviceType> matrixMatrixMul(const Tensor<T, DeviceType>& a, const Tensor<T, DeviceType>& b) {
        if (a.getShape()[1] != b.getShape()[0]) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication.");
        }

        int m = a.getShape()[0];
        int n = b.getShape()[1];
        int k = a.getShape()[1];

        Tensor<T, DeviceType> result({m, n}, T(0));

        const T* aData = a.getData().data();
        const T* bData = b.getData().data();
        T* resultData = result.getData().data();

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; i += TILE_SIZE) {
            for (int j = 0; j < n; j += TILE_SIZE) {
                for (int l = 0; l < k; l += TILE_SIZE) {
                    int max_i = std::min(i + TILE_SIZE, m);
                    int max_j = std::min(j + TILE_SIZE, n);
                    int max_l = std::min(l + TILE_SIZE, k);

                    for (int ii = i; ii < max_i; ++ii) {
                        for (int jj = j; jj < max_j; ++jj) {
                            T sum = 0;
                            #pragma omp simd reduction(+:sum)
                            for (int ll = l; ll < max_l; ++ll) {
                                sum += aData[ii * k + ll] * bData[ll * n + jj];
                            }
                            resultData[ii * n + jj] += sum;
                        }
                    }
                }
            }
        }

        return result;
    }

    static Tensor<T, DeviceType> batchedMatmul(const Tensor<T, DeviceType>& a, const Tensor<T, DeviceType>& b) {
        auto shapeA = a.getShape();
        auto shapeB = b.getShape();

        if (shapeA.rank() == 1) {
            shapeA = Shape({1}).concat(shapeA);
        }

        if (shapeB.rank() == 1) {
            shapeB = shapeB.concat(Shape({1}));
        }

        auto batchDimsA = Shape(std::vector<int>(shapeA.begin(), shapeA.end() - 2));
        auto batchDimsB = Shape(std::vector<int>(shapeB.begin(), shapeB.end() - 2));

        auto batchShape = ShapeOperations::broadcastShapes(batchDimsA, batchDimsB);

        auto resultShape = batchShape.concat(Shape({shapeA[shapeA.rank() - 2], shapeB[shapeB.rank() - 1]}));
        std::vector<int> dimensions = resultShape.getDimensions();
        Tensor<T, DeviceType> result = Tensor::zeros(resultShape);

        auto flattenedBatchSize = batchShape.size();
        #pragma omp parallel for
        for (int batch = 0; batch < flattenedBatchSize; ++batch) {
            
            // TODO: Implement Slice extraction
            auto aSlice = extractSlice(a, batch, batchShape, batchDimsA);
            auto bSlice = extractSlice(b, batch, batchShape, batchDimsB);

            // Perform matrix multiplication on the slices
            auto batchResult = matrixMatrixMul(aSlice, bSlice);

            // Not complete, need to implement the insertion of the batch result into the final result
        }

        // Remove prepended/appended dimensions if necessary
        if (a.getShape().rank() == 1) {
            dimensions.pop(dimensions.size()-1);
            result.reshape(dimensions);
        }
        if (b.getShape().rank() == 1) {
            dimensions.pop(dimensions.size()-2)
            result.reshape(dimensions);
        }

        return result;
    }

};

}; // namespace smart_dnn

#endif // ADVANCED_TENSOR_OPERATIONS_HPP
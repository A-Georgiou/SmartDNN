#ifndef ADVANCED_TENSOR_OPERATIONS_HPP
#define ADVANCED_TENSOR_OPERATIONS_HPP

#include "TensorData.hpp"
#include "DeviceTypes.hpp"
#include "Tensor.hpp"
#include "../Shape/ShapeOperations.hpp"
#include "../Tensor/TensorOperations.hpp"

namespace smart_dnn {

// Forward declaration of the primary template
template <typename T, typename DeviceType=CPUDevice>
class AdvancedTensorOperations;

// Specialization for CPUDevice
template <typename T>
class AdvancedTensorOperations<T, CPUDevice> {
    public:
    AdvancedTensorOperations() = delete;

    static Tensor<T> apply(const Tensor<T>& tensor, std::function<T(T)> func) {
        return TensorOperations<T, CPUDevice>::apply(tensor.getData(), func);
    }

    static Tensor<T> sum(const Tensor<T>& tensor) {
        return TensorOperations<T, CPUDevice>::sum(tensor.getData());
    }

    static Tensor<T> reshape(const Tensor<T>& tensor, const Shape& newShape) {
        Tensor<T> result = tensor;
        result.reshape(newShape);
        return result;
    }

    static Tensor<T> transpose(const Tensor<T>& tensor, int dim0, int dim1) {
        auto shape = tensor.getShape();
        auto rank = shape.rank();

        if (dim0 >= rank || dim1 >= rank || dim0 < 0 || dim1 < 0) {
            throw std::invalid_argument("Invalid dimensions for transpose.");
        }

        std::vector<int> newDimensions = shape.getDimensions();
        std::swap(newDimensions[dim0], newDimensions[dim1]);
        Shape newShape(newDimensions);

        TensorData<T, CPUDevice> resultData = tensor.getData();
        resultData.reshape(newShape);
        return resultData;
    }

    static Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
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

    static Tensor<T> dotProduct(const Tensor<T>& a, const Tensor<T>& b) {
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

        return Tensor<T>({1}, {result});
    }

    static Tensor<T> matrixVectorMul(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.getShape()[1] != b.getShape()[0]) {
            throw std::invalid_argument("Matrix and vector dimensions must match for multiplication.");
        }

        int m = a.getShape()[0];
        int n = a.getShape()[1];
        Tensor<T> result({m}, T(0));

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

    static Tensor<T> matrixMatrixMul(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.getShape()[1] != b.getShape()[0]) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication.");
        }

        int m = a.getShape()[0];
        int n = b.getShape()[1];
        int k = a.getShape()[1];

        Tensor<T> result({m, n}, T(0));

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

    static Tensor<T> batchedMatmul(const Tensor<T>& a, const Tensor<T>& b) {
        auto shapeA = a.getShape();
        auto shapeB = b.getShape();

        if (shapeA.rank() == 1) {
            shapeA = ShapeOperations::concat(Shape({1}), shapeA);
        }

        if (shapeB.rank() == 1) {
            shapeB = ShapeOperations::concat(shapeB, Shape({1}));
        }

        auto batchDimsA = Shape(std::vector<int>(shapeA.begin(), shapeA.end() - 2));
        auto batchDimsB = Shape(std::vector<int>(shapeB.begin(), shapeB.end() - 2));

        auto batchShape = ShapeOperations::broadcastShapes(batchDimsA, batchDimsB);

        auto resultShape = ShapeOperations::concat(batchShape, Shape({shapeA[shapeA.rank() - 2], shapeB[shapeB.rank() - 1]}));
        std::vector<int> dimensions = resultShape.getDimensions();
        Tensor<T> result = Tensor<T>::zeros(resultShape);

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
            dimensions.erase(dimensions.begin()+(dimensions.size()-1));
            result.reshape(dimensions);
        }
        if (b.getShape().rank() == 1) {
            dimensions.erase(dimensions.begin()+(dimensions.size()-2));
            result.reshape(dimensions);
        }

        return result;
    }
private:
    static Tensor<T> extractSlice(const Tensor<T>& tensor, int batchIndex, const Shape& batchShape, const Shape& originalBatchDims) {
        // Get the rank and shape of the tensor
        auto rank = tensor.getShape().rank();
        auto fullShape = tensor.getShape().getDimensions();

        std::vector<int> startIndices(rank, 0);
        std::vector<int> endIndices = fullShape;

        auto strides = tensor.getShape().getStride();

        int offset = 0;
        int batchSize = 1;
        for (int i = 0; i < originalBatchDims.rank(); ++i) {
            int dimSize = batchShape[i];
            int dimIndex = (batchIndex / batchSize) % dimSize;
            offset += dimIndex * strides[i];
            batchSize *= dimSize;
        }

        std::vector<int> sliceShape(fullShape.begin() + originalBatchDims.rank(), fullShape.end());

        Tensor<T> sliceTensor(Shape(sliceShape), tensor.getData().begin() + offset);

        return sliceTensor;
    }
};

}; // namespace smart_dnn

#endif // ADVANCED_TENSOR_OPERATIONS_HPP
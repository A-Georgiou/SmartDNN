#ifndef ADVANCED_TENSOR_OPERATIONS_HPP
#define ADVANCED_TENSOR_OPERATIONS_HPP

#include "smart_dnn/Tensor/TensorData.hpp"
#include "smart_dnn/Tensor/DeviceTypes.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Shape/ShapeOperations.hpp"
#include "smart_dnn/Tensor/SliceView.hpp"
#include "smart_dnn/Tensor/TensorOperations.hpp"
#include "smart_dnn/Tensor/BroadcastView.hpp"
#include <functional>

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

    template <typename Func>
    static Tensor<T> applyPair(const Tensor<T>& a, const Tensor<T>& b, Func f) {
        if (a.getShape() != b.getShape()) {
            throw std::invalid_argument("Tensors must have the same shape");
        }
        Tensor<T> result(a.getShape());
        const T* aData = a.getData().data();
        const T* bData = b.getData().data();
        T* resultData = result.getData().data();
        int size = a.getShape().size();

        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            resultData[i] = f(aData[i], bData[i]);
        }

        return result;
    }

    static Tensor<T> sum(const Tensor<T>& tensor) {
        return TensorOperations<T, CPUDevice>::sum(tensor.getData());
    }

    static Tensor<T> sum(const Tensor<T>& tensor, size_t axis) {
        return TensorOperations<T, CPUDevice>::sum(tensor.getData(), axis);
    }

    static Tensor<T> sum(const Tensor<T>& tensor, const std::vector<size_t>& axes) {
        const auto& shape = tensor.getShape();
        std::vector<int> newShape = shape.getDimensions();
        std::vector<bool> axesToSum(shape.rank(), false);
        
        for (size_t axis : axes) {
            if (axis < 0 || axis >= shape.rank()) {
                throw std::runtime_error("Invalid axis for sum operation, axis: " + std::to_string(axis) + ", tensor rank: " + std::to_string(tensor.getShape().rank()));
            }
            axesToSum[axis] = true;
            newShape[axis] = 1;
        }

        Tensor<T> result(Shape(newShape), T(0));
        const T* inputData = tensor.getData().data();
        T* resultData = result.getData().data();

        std::vector<int> inputIndices(shape.rank(), 0);
        std::vector<int> resultIndices(shape.rank(), 0);
        
        do {
            int inputIndex = computeFlatIndex(shape, inputIndices);
            int resultIndex = computeFlatIndex(result.getShape(), resultIndices);
            
            resultData[resultIndex] += inputData[inputIndex];
        } while (incrementIndices(inputIndices, shape, &axesToSum, &resultIndices));

        // Remove summed dimensions
        std::vector<int> finalShape;
        for (size_t i = 0; i < newShape.size(); ++i) {
            if (newShape[i] != 1 || !axesToSum[i]) {
                finalShape.push_back(newShape[i]);
            }
        }

        result.reshape(Shape(finalShape));

        return result;
    }

    // Reciprocal function: calculates the reciprocal of each element in the tensor
    // Reciprocol is defined as: f(x) = 1 / x if abs(x) > epsilon, else 1 / epsilon
    static Tensor<T> reciprocal(const Tensor<T>& tensor, T epsilon = T(1e-12)) {
        return apply(tensor, [epsilon](T x) { return (std::abs(x) > epsilon) ? (T(1) / x) : (T(1) / epsilon); });
    }

    static Tensor<T> mean(const Tensor<T>& tensor, const std::vector<size_t>& axes) {
        if (axes.empty() || axes.size() == tensor.getShape().rank()) {
            return mean(tensor);
        }

        const auto& shape = tensor.getShape();
        Tensor<T> result = tensor;
        int totalElements = 1;

        std::vector<int> newShape(shape.rank(), 1);
        for (size_t i = 0; i < shape.rank(); ++i) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
                newShape[i] = shape[i];
            } else {
                totalElements *= shape[i];
            }
        }

        // Sort axes in descending order
        std::vector<size_t> sortedAxes = axes;
        std::sort(sortedAxes.begin(), sortedAxes.end(), std::greater<size_t>());

        for (size_t axis : sortedAxes) {
            if (axis >= result.getShape().rank() || axis < 0) {
                throw std::invalid_argument("Invalid axis for mean calculation");
            }
            result = sum(result, axis);
        }

        // Divide the sum by the total number of elements to get the mean
        result = result * (T(1) / T(totalElements));
        result = reshape(result, Shape(newShape));

        return result;
    }


    // Mean function: calculates the mean for the entire tensor (all elements)
    static Tensor<T> mean(const Tensor<T>& tensor) {
        return sum(tensor) * (T(1) / T(tensor.getShape().size()));
    }

    // Variance function: calculates the variance between tensors (all elements) along the specified axes
    static Tensor<T> variance(const Tensor<T>& tensor, const Tensor<T>& meanTensor, const std::vector<size_t>& axes) {
        const auto& shape = tensor.getShape();
        const auto& meanShape = meanTensor.getShape();

        int totalElements = 1;
        for (size_t axis : axes) {
            if (axis < 0 || axis >= shape.rank()) {
                throw std::runtime_error("Invalid axis for variance calculation: " + std::to_string(axis));
            }
            totalElements *= shape[axis];
        }
        // Create a BroadcastView of the mean tensor
        BroadcastView<T, CPUDevice> broadcastedMean(meanTensor.getData(), shape);

        // Compute the difference and square it
        Tensor<T> squaredDiff(shape);
        auto tensorIt = tensor.getData().begin();
        auto broadcastIt = broadcastedMean.begin();
        auto squaredDiffIt = squaredDiff.getData().begin();

        while (tensorIt != tensor.getData().end()) {
            *squaredDiffIt = (*tensorIt - *broadcastIt) * (*tensorIt - *broadcastIt);
            ++tensorIt;
            ++broadcastIt;
            ++squaredDiffIt;
        }

        Tensor<T> summedSquaredDiff = sum(squaredDiff, axes);

        // Ensure the result has the same shape as the mean tensor
        Tensor<T> result(meanShape);
        for (size_t i = 0; i < result.getData().size(); ++i) {
            result.getData()[i] = summedSquaredDiff.getData()[i] / T(totalElements);
        }

        return result;
    }

    // Variance function: calculates the variance for the entire tensor (all elements)
    // The variance is the mean of the squared differences from the mean
    static Tensor<T> variance(const Tensor<T>& tensor, const Tensor<T>& meanTensor) {
        Tensor<T> diff = tensor - meanTensor;
        return sum(diff * diff) * (T(1) / T(tensor.getShape().size()));
    }

    static Tensor<T> reshape(const Tensor<T>& tensor, const Shape& newShape) {
        Tensor<T> result = tensor;
        result.reshape(newShape);
        return result;
    }

    static Tensor<T> transpose(const Tensor<T>& tensor, size_t dim0, size_t dim1) {
        const Shape& shape = tensor.getShape();
        const size_t rank = shape.rank();

        if (dim0 >= rank || dim1 >= rank) {
            throw std::invalid_argument("Invalid dimensions for transpose.");
        }

        std::vector<int> newDimensions = shape.getDimensions();
        std::swap(newDimensions[dim0], newDimensions[dim1]);
        Shape newShape(newDimensions);
        Tensor<T> result(newShape);

        // Input element at index (.., i, .., j, ..) lands at output index (.., j, .., i, ..).
        std::vector<size_t> outStride = newShape.getStride();
        std::swap(outStride[dim0], outStride[dim1]);

        const T* in = tensor.getData().begin();
        T* out = result.getData().begin();
        std::vector<int> index(rank, 0);
        for (size_t i = 0; i < shape.size(); ++i) {
            size_t offset = 0;
            for (size_t d = 0; d < rank; ++d) {
                offset += static_cast<size_t>(index[d]) * outStride[d];
            }
            out[offset] = in[i];
            for (int d = static_cast<int>(rank) - 1; d >= 0 && ++index[d] == shape[d]; --d) {
                index[d] = 0;
            }
        }
        return result;
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

    /*
    
    Dot Product:
    ------------
    Input: Tensor a (n), Tensor b (n)
    Output: Tensor result (1)

    For two vectors a and b, the dot product is the sum of the element-wise product of the two vectors.
    The output is a scalar value.
    
    */
    static Tensor<T> dotProduct(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.getShape()[0] != b.getShape()[0]) {
            throw std::invalid_argument("Vector dimensions must match for dot product. Mismatch in dimensions: " + a.getShape().toString() + " and " + b.getShape().toString());
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

    /*
    
    Dot Product Slice:
    -----------------------------

    Input: SliceView a, SliceView b
    Output: Scalar result

    For two slices a and b, the dot product is the sum of the element-wise product of the two slices.
    The output is a scalar value.

    */

    static T dotProductSlice(const SliceView<T>& a, const SliceView<T>& b){
        if (a.shape() != b.shape()){
            throw std::invalid_argument("Slice dimensions must match for dot product. Mismatch in dimensions: " + a.shape().toString() + " and " + b.shape().toString());
        }

        T result = T(0);

        auto a_it = a.begin();
        auto b_it = b.begin();

        while(a_it != a.end()){
            result += *a_it * *b_it;
            ++a_it;
            ++b_it;
        }

        return result;
    }

    /*
    
    Matrix Vector Product:
    -----------------------------
    Input: Tensor a (m x n), Tensor b (n)
    Output: Tensor result (m)

    For each row in a, multiply the row with the vector b and sum the results to get the result vector.
    The output is a vector of size m.
    
    */
    static Tensor<T> matrixVectorMul(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.getShape()[1] != b.getShape()[0]) {
            throw std::invalid_argument("Matrix and vector dimensions must match for multiplication. Mismatch in dimensions: " + a.getShape().toString() + " and " + b.getShape().toString());
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


    /*
    
    Matrix Matrix Product:
    -----------------------------
    Input: Tensor a (m x k), Tensor b (k x n)
    Output: Tensor result (m x n)

    Each value i,j is equal to the dot product of the i-th row of a and the j-th column of b.
    
    */
    static Tensor<T> matrixMatrixMul(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.getShape()[1] != b.getShape()[0]) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication. Mismatch in dimensions: " 
                + a.getShape().toString() + " and " + b.getShape().toString());
        }

        int m = a.getShape()[0];
        int n = b.getShape()[1];

        Tensor<T> result({m, n}, T(0));
        multiplyInto(a.getData().data(), b.getData().data(), result.getData().data(), m, a.getShape()[1], n);
        return result;
    }

    /*

    Matrix Multiply Kernel:
    -----------------------------
    Adds a (m x k) times b (k x n) into c (m x n); all three are contiguous row-major blocks.
    Row i of c is accumulated as a[i][p] * (row p of b), so the inner loop runs over contiguous memory.

    */
    static void multiplyInto(const T* a, const T* b, T* c, size_t m, size_t k, size_t n) {
        // A single output column is m dot products; accumulating each in a register beats the row loop.
        if (n == 1) {
            for (size_t i = 0; i < m; i++) {
                T sum = 0;
                for (size_t p = 0; p < k; p++) {
                    sum += a[i * k + p] * b[p];
                }
                c[i] += sum;
            }
            return;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < m; i++) {
            T* cRow = c + i * n;
            for (size_t p = 0; p < k; p++) {
                const T aValue = a[i * k + p];
                const T* bRow = b + p * n;
                for (size_t j = 0; j < n; j++) {
                    cRow[j] += aValue * bRow[j];
                }
            }
        }
    }
    /*
    
    Matrix Matrix Prduct Slice:
    -----------------------------

    Input: SliceView a (m x k), SliceView b (k x n)
    Output: Tensor result (m x n)

    Each value i,j is equal to the dot product of the i-th row of a and the j-th column of b.
    
    */

   static Tensor<T> matrixMatrixMulSlice(const SliceView<T>& a, const SliceView<T>& b) {
        const auto& shapeA = a.shape();
        const auto& shapeB = b.shape();

        if (shapeA.rank() != 2 || shapeB.rank() != 2 || shapeA[1] != shapeB[0]) {
            throw std::invalid_argument("Invalid slice shapes for matrix multiplication. Got shapes: " 
                + shapeA.toString() + " and " + shapeB.toString());
        }

        int m = shapeA[0];
        int n = shapeB[1];
        int k = shapeA[1];

        Tensor<T> result({m, n}, T(0));
        T* resultData = result.getData().data();

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                T sum = 0;
                for (int p = 0; p < k; ++p) {
                    sum += a[{i, p}] * b[{p, j}];
                }
                resultData[i * n + j] = sum;
            }
        }

        return result;
    }


   /*
   
    Batched Matrix Multiplication:
    -----------------------------

    Input: Tensor a (... x m x k), Tensor b (... x k x n), at least one of rank three or more
    Output: Tensor result (... x m x n)

    Batch dimensions broadcast like NumPy: aligned from the right, each pair equal or 1.
    Each batch is one multiplyInto call, and a 2-D b multiplies every row of a in a single call.
   
   */

    static Tensor<T> batchedMatmul(const Tensor<T>& a, const Tensor<T>& b) {
        const Shape& shapeA = a.getShape();
        const Shape& shapeB = b.getShape();
        const int rankA = shapeA.rank();
        const int rankB = shapeB.rank();
        if (rankA < 2 || rankB < 2 || shapeA[rankA - 1] != shapeB[rankB - 2]) {
            throw std::invalid_argument("Incompatible dimensions for matrix multiplication: " + shapeA.toString() + " and " + shapeB.toString());
        }

        const int m = shapeA[rankA - 2];
        const int k = shapeA[rankA - 1];
        const int n = shapeB[rankB - 1];

        if (rankB == 2) {
            std::vector<int> dims = shapeA.getDimensions();
            dims.back() = n;
            Tensor<T> result(Shape(dims), T(0));
            multiplyInto(a.getData().data(), b.getData().data(), result.getData().data(), shapeA.size() / k, k, n);
            return result;
        }

        const int batchRank = std::max(rankA, rankB) - 2;
        std::vector<int> dims(batchRank);
        std::vector<size_t> stepA(batchRank), stepB(batchRank);  // Offset change per batch index; 0 on broadcast axes.
        size_t strideA = static_cast<size_t>(m) * k;
        size_t strideB = static_cast<size_t>(k) * n;
        size_t batches = 1;
        for (int d = batchRank - 1; d >= 0; --d) {
            const int axisA = d - batchRank + rankA - 2;
            const int axisB = d - batchRank + rankB - 2;
            const int sizeA = axisA >= 0 ? shapeA[axisA] : 1;
            const int sizeB = axisB >= 0 ? shapeB[axisB] : 1;
            if (sizeA != sizeB && sizeA != 1 && sizeB != 1) {
                throw std::invalid_argument("Batch dimensions must match or be 1: " + shapeA.toString() + " and " + shapeB.toString());
            }
            dims[d] = std::max(sizeA, sizeB);
            stepA[d] = sizeA == 1 ? 0 : strideA;
            stepB[d] = sizeB == 1 ? 0 : strideB;
            strideA *= sizeA;
            strideB *= sizeB;
            batches *= dims[d];
        }

        dims.push_back(m);
        dims.push_back(n);
        Tensor<T> result(Shape(dims), T(0));
        const T* aData = a.getData().data();
        const T* bData = b.getData().data();
        T* resultData = result.getData().data();

        std::vector<int> index(batchRank, 0);
        size_t offsetA = 0;
        size_t offsetB = 0;
        for (size_t batch = 0; batch < batches; ++batch) {
            multiplyInto(aData + offsetA, bData + offsetB, resultData + batch * m * n, m, k, n);
            for (int d = batchRank - 1; d >= 0; --d) {
                offsetA += stepA[d];
                offsetB += stepB[d];
                if (++index[d] < dims[d]) {
                    break;
                }
                offsetA -= stepA[d] * dims[d];
                offsetB -= stepB[d] * dims[d];
                index[d] = 0;
            }
        }

        return result;
    }

    template<typename ShapeType>
    static bool incrementIndices(std::vector<int>& indices, 
                                 const ShapeType& shape, 
                                 const std::vector<bool>* axesToSum = nullptr, 
                                 std::vector<int>* resultIndices = nullptr) {
        for (int i = indices.size() - 1; i >= 0; --i) {
            ++indices[i];
            if (indices[i] < getShapeElement(shape, i)) {
                if (axesToSum && resultIndices && !(*axesToSum)[i]) {
                    ++(*resultIndices)[i];
                }
                return true;
            }
            indices[i] = 0;
            if (axesToSum && resultIndices && !(*axesToSum)[i]) {
                (*resultIndices)[i] = 0;
            }
        }
        return false;
    }

    template<typename ShapeType>
    static int getShapeElement(const ShapeType& shape, int index) {
        if constexpr (std::is_same_v<ShapeType, Shape>) {
            return shape[index];
        } else {
            return shape.at(index);
        }
    }
    
};

}; // namespace smart_dnn

#endif // ADVANCED_TENSOR_OPERATIONS_HPP
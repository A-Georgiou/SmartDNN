#ifndef ADVANCED_TENSOR_OPERATIONS_HPP
#define ADVANCED_TENSOR_OPERATIONS_HPP

#include <functional>
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include "smart_dnn/tensor/Backend/Default/BroadcastView.hpp"

namespace sdnn {

// Specialization for CPUDevice
class AdvancedTensorOperations {
    public:
    AdvancedTensorOperations() = delete;

    // Reciprocal function: calculates the reciprocal of each element in the tensor
    // Reciprocol is defined as: f(x) = 1 / x if abs(x) > epsilon, else 1 / epsilon
    static Tensor reciprocal(const Tensor& tensor, double epsilon = 1e-12) {
        return apply(tensor, [epsilon](double x) { return (std::abs(x) > epsilon) ? (1 / x) : (1 / epsilon); });
    }

    static Tensor mean(const Tensor& tensor, const std::vector<int>& axes) {
        if (axes.empty() || axes.size() == tensor.shape().rank()) {
            return mean(tensor);
        }

        const auto& shape = tensor.shape();
        Tensor result = tensor;
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
        std::vector<int> sortedAxes = axes;
        std::sort(sortedAxes.begin(), sortedAxes.end(), std::greater<int>());

        for (int axis : sortedAxes) {
            if (axis >= result.shape().rank() || axis < 0) {
                throw std::invalid_argument("Invalid axis for mean calculation");
            }
            result = sum(result, {axis});
        }

        // Divide the sum by the total number of elements to get the mean
        result = result * (1 / totalElements);
        result = reshape(result, Shape(newShape));

        return result;
    }


    // Mean function: calculates the mean for the entire tensor (all elements)
    static Tensor mean(const Tensor& tensor) {
        return sum(tensor) * (1 / tensor.shape().size());
    }

    /*

    // Variance function: calculates the variance between tensors (all elements) along the specified axes
    static Tensor variance(const Tensor& tensor, const Tensor& meanTensor, const std::vector<size_t>& axes) {
        const auto& shape = tensor.shape();
        const auto& meanShape = meanTensor.shape();

        int totalElements = 1;
        for (size_t axis : axes) {
            if (axis < 0 || axis >= shape.rank()) {
                throw std::runtime_error("Invalid axis for variance calculation: " + std::to_string(axis));
            }
            totalElements *= shape[axis];
        }
        // Create a BroadcastView of the mean tensor
        BroadcastView broadcastedMean(meanTensor.getData(), shape);

        // Compute the difference and square it
        Tensor squaredDiff(shape);
        auto tensorIt = tensor.getData().begin();
        auto broadcastIt = broadcastedMean.begin();
        auto squaredDiffIt = squaredDiff.getData().begin();

        while (tensorIt != tensor.getData().end()) {
            *squaredDiffIt = (*tensorIt - *broadcastIt) * (*tensorIt - *broadcastIt);
            ++tensorIt;
            ++broadcastIt;
            ++squaredDiffIt;
        }

        Tensor summedSquaredDiff = sum(squaredDiff, axes);

        // Ensure the result has the same shape as the mean tensor
        Tensor result(meanShape);
        for (size_t i = 0; i < result.getData().size(); ++i) {
            result.getData()[i] = summedSquaredDiff.getData()[i] / T(totalElements);
        }

        return result;
    }

    // Variance function: calculates the variance for the entire tensor (all elements)
    // The variance is the mean of the squared differences from the mean
    static Tensor variance(const Tensor& tensor, const Tensor& meanTensor) {
        Tensor diff = tensor - meanTensor;
        return sum(diff * diff) * (T(1) / T(tensor.getShape().size()));
    }

    static Tensor reshape(const Tensor& tensor, const Shape& newShape) {
        Tensor result = tensor;
        result.reshape(newShape);
        return result;
    }

    static Tensor transpose(const Tensor& tensor, size_t dim0, size_t dim1) {
        auto shape = tensor.getShape();
        auto rank = shape.rank();

        if (dim0 >= rank || dim1 >= rank) {
            throw std::invalid_argument("Invalid dimensions for transpose.");
        }

        std::vector<int> newDimensions = shape.getDimensions();
        std::swap(newDimensions[dim0], newDimensions[dim1]);
        Shape newShape(newDimensions);

        Tensor result(newShape);
        
        // Create a mapping from old indices to new indices
        std::vector<size_t> oldToNew(rank);
        for (size_t i = 0; i < rank; ++i) {
            oldToNew[i] = i;
        }
        std::swap(oldToNew[dim0], oldToNew[dim1]);

        // Iterate through all elements and place them in their new positions
        std::vector<int> oldIndices(rank, 0);
        std::vector<int> newIndices(rank, 0);
        size_t totalSize = shape.size();

        #pragma omp parallel for
        for (size_t i = 0; i < totalSize; ++i) {
            // Calculate old indices
            size_t temp = i;
            for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
                if (shape[d] == 0) {
                    throw std::runtime_error("Invalid shape: dimension size is zero");
                }
                oldIndices[d] = static_cast<int>(temp % static_cast<size_t>(shape[d]));
                temp /= static_cast<size_t>(shape[d]);
            }

            // Map to new indices
            for (size_t d = 0; d < rank; ++d) {
                newIndices[d] = oldIndices[oldToNew[d]];
            }

            // Set the value in the new tensor
            result.at(newIndices) = tensor.at(oldIndices);
        }

        return result;
    }

    */

    static Tensor matmul(const Tensor& a, const Tensor& b) {
        const auto& shapeA = a.shape();
        const auto& shapeB = b.shape();

        if (shapeA.rank() == 1 && shapeB.rank() == 1) {
            return dotProduct(a, b);
        } else if (shapeA.rank() == 2 && shapeB.rank() == 2) {
            return matrixMatrixMul(a, b);
        } else if (shapeA.rank() == 1 && shapeB.rank() == 2) {
            auto a_reshaped = reshape(a, {1, shapeA[0]});
            auto result = matrixMatrixMul(a_reshaped, b);
            return reshape(result, {result.shape()[1]});
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
    static Tensor dotProduct(const Tensor& a, const Tensor& b) {
        if (a.shape()[0] != b.shape()[0]) {
            throw std::invalid_argument("Vector dimensions must match for dot product.");
        }

        double result = 0;
        for (size_t i = 0; i < a.shape()[0]; ++i) {
            result += a.tensorImpl_->getValueAsDouble(i) * b.tensorImpl_->getValueAsDouble(i);
        }

        return Tensor(createTensorAdapter(Shape({1}), &result, a.type()));
    }

    /*
    
    Matrix Vector Product:
    -----------------------------
    Input: Tensor a (m x n), Tensor b (n)
    Output: Tensor result (m)

    For each row in a, multiply the row with the vector b and sum the results to get the result vector.
    The output is a vector of size m.
    
    */
    static Tensor matrixVectorMul(const Tensor& a, const Tensor& b) {
        const auto& shapeA = a.shape();
        const auto& shapeB = b.shape();

        if (shapeA[1] != shapeB[0]) {
            throw std::invalid_argument("Matrix and vector dimensions must match for multiplication.");
        }

        Shape resultShape({shapeA[0]});
        auto result = createTensorAdapter(resultShape, a.type());

        for (size_t i = 0; i < shapeA[0]; ++i) {
            double sum = 0;
            for (size_t j = 0; j < shapeA[1]; ++j) {
                sum += a.tensorImpl_->getValueAsDouble(i * shapeA[1] + j) * b.tensorImpl_->getValueAsDouble(j);
            }
            result->setValueFromDouble(i, sum);
        }

        return Tensor(std::move(result));
    }


    /*
    
    Matrix Matrix Product:
    -----------------------------
    Input: Tensor a (m x k), Tensor b (k x n)
    Output: Tensor result (m x n)

    Each value i,j is equal to the dot product of the i-th row of a and the j-th column of b.
    For this approach we slice our data into rows and columns and calculate the dot product of the slices.
    
    */
    static Tensor matrixMatrixMul(const Tensor& a, const Tensor& b) {
        const auto& shapeA = a.shape();
        const auto& shapeB = b.shape();

        if (shapeA[1] != shapeB[0]) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication.");
        }

        Shape resultShape({shapeA[0], shapeB[1]});
        auto result = createTensorAdapter(resultShape, a.type());

        for (size_t i = 0; i < shapeA[0]; ++i) {
            for (size_t j = 0; j < shapeB[1]; ++j) {
                double sum = 0;
                for (size_t k = 0; k < shapeA[1]; ++k) {
                    sum += a.tensorImpl_->getValueAsDouble(i * shapeA[1] + k) * 
                           b.tensorImpl_->getValueAsDouble(k * shapeB[1] + j);
                }
                result->setValueFromDouble(i * shapeB[1] + j, sum);
            }
        }

        return Tensor(std::move(result));
    }

   /*
   
    Batched Matrix Multiplication:
    -----------------------------

    Input: Tensor a (b x m x k), Tensor b (b x k x n)
    Output: Tensor result (b x m x n)

    For each batch, perform matrix multiplication on the corresponding slices of a and b.
    The output is a tensor of shape (b x m x n).
   
   */

    static Tensor batchedMatmul(const Tensor& a, const Tensor& b) {
        const auto& shapeA = a.shape();
        const auto& shapeB = b.shape();

        // Handle 3D x 2D case
        if (shapeA.rank() == 3 && shapeB.rank() == 2) {
            return matmul3D2D(a, b);
        }

        // Handle 2D x 3D case
        if (shapeA.rank() == 2 && shapeB.rank() == 3) {
            return matmul2D3D(a, b);
        }

        if (shapeA.rank() != shapeB.rank()) {
            throw std::invalid_argument("Tensors must have the same rank for batched matmul");
        }

        int rank = shapeA.rank();
        if (rank < 3) {
            throw std::invalid_argument("Tensors must have at least 3 dimensions for batched matmul");
        }

        // Check if the last two dimensions are compatible for matrix multiplication
        if (shapeA[rank - 1] != shapeB[rank - 2]) {
            throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
        }

        // Calculate the shape of the result tensor
        std::vector<int> resultShape;
        for (int i = 0; i < rank - 2; ++i) {
            resultShape.push_back(std::max(shapeA[i], shapeB[i]));
        }
        resultShape.push_back(shapeA[rank - 2]);
        resultShape.push_back(shapeB[rank - 1]);
        
        auto result = createTensorAdapter(Shape(resultShape), a.type());

        // Iterate over all batch dimensions
        std::vector<size_t> batchIndices(rank - 2, 0);
        do {
            // Perform matrix multiplication for this batch
            for (int i = 0; i < shapeA[rank - 2]; ++i) {
                for (int j = 0; j < shapeB[rank - 1]; ++j) {
                    double sum = 0;
                    for (int k = 0; k < shapeA[rank - 1]; ++k) {
                        std::vector<size_t> aIndices = batchIndices;
                        aIndices.push_back(i);
                        aIndices.push_back(k);

                        std::vector<size_t> bIndices = batchIndices;
                        bIndices.push_back(k);
                        bIndices.push_back(j);

                        sum += a.tensorImpl_->getValueAsDouble(computeFlatIndex(shapeA, aIndices)) * 
                               b.tensorImpl_->getValueAsDouble(computeFlatIndex(shapeB, bIndices));
                    }


                    std::vector<size_t> resultIndices = batchIndices;
                    resultIndices.push_back(i);
                    resultIndices.push_back(j);
                    result->setValueFromDouble(computeFlatIndex(Shape(resultShape), resultIndices), sum);
                }
            }
        } while (incrementIndices(batchIndices, resultShape));

        return Tensor(std::move(result));
    }


    static Tensor matmul2D3D(const Tensor& a, const Tensor& b) {
        const auto& shapeA = a.shape();
        const auto& shapeB = b.shape();

        if (shapeA.rank() != 2 || shapeB.rank() != 3 || shapeA[1] != shapeB[1]) {
            throw std::invalid_argument("Invalid shapes for matmul 2D x 3D. Got shapes: " 
                + shapeA.toString() + " and " + shapeB.toString());
        }

        int outFeatures = shapeA[0];
        int inFeatures = shapeA[1];
        int batchSize = shapeB[0];
        int sequenceLength = shapeB[2];

        auto result = createTensorAdapter(Shape({batchSize, outFeatures, sequenceLength}), a.type());

        for (int n = 0; n < batchSize; ++n) {
            for (int i = 0; i < outFeatures; ++i) {
                for (int j = 0; j < sequenceLength; ++j) {
                    double sum = 0;
                    for (int k = 0; k < inFeatures; ++k) {
                        sum += a.tensorImpl_->getValueAsDouble(i * inFeatures + k) * 
                               b.tensorImpl_->getValueAsDouble((n * inFeatures * sequenceLength) + (k * sequenceLength) + j);
                    }
                    result->setValueFromDouble((n * outFeatures * sequenceLength) + (i * sequenceLength) + j, sum);
                }
            }
        }

        return Tensor(std::move(result));
    }

    static Tensor matmul3D2D(const Tensor& a, const Tensor& b) {
        const auto& shapeA = a.shape();
        const auto& shapeB = b.shape();

        if (shapeA.rank() != 3 || shapeB.rank() != 2 || shapeA[2] != shapeB[0]) {
            throw std::invalid_argument("Invalid shapes for matmul 3D x 2D. Got shapes: " 
                + shapeA.toString() + " and " + shapeB.toString());
        }

        int batchSize = shapeA[0];
        int m = shapeA[1];
        int n = shapeA[2];
        int p = shapeB[1];

        auto result = createTensorAdapter(Shape({batchSize, m, p}), a.type());

        for (int batch = 0; batch < batchSize; ++batch) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < p; ++j) {
                    double sum = 0;
                    for (int k = 0; k < n; ++k) {
                        double aVal = a.tensorImpl_->getValueAsDouble((batch * m * n) + (i * n) + k);
                        double bVal = b.tensorImpl_->getValueAsDouble(k * p + j);
                        sum += aVal * bVal;
                    }
                    result->setValueFromDouble((batch * m * p) + (i * p) + j, sum);
                }
            }
        }

        return Tensor(std::move(result));
    }

    template<typename ShapeType>
    static bool incrementIndices(std::vector<size_t>& indices, 
                                 const ShapeType& shape, 
                                 const std::vector<bool>* axesToSum = nullptr, 
                                 std::vector<int>* resultIndices = nullptr) {
        for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
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
    static int getShapeElement(const ShapeType& shape, size_t index) {
        if constexpr (std::is_same_v<ShapeType, Shape>) {
            return shape[index];
        } else {
            return shape.at(index);
        }
    }
    
};

}; // namespace sdnn

#endif // ADVANCED_TENSOR_OPERATIONS_HPP
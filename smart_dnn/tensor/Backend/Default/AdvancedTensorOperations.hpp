#ifndef ADVANCED_TENSOR_OPERATIONS_HPP
#define ADVANCED_TENSOR_OPERATIONS_HPP

#include <functional>
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/Shape/ShapeOperations.hpp"
#include "smart_dnn/tensor/Backend/Default/BroadcastView.hpp"

namespace sdnn {

// Specialization for CPUDevice
class AdvancedTensorOperations {
    public:
    AdvancedTensorOperations() = delete;

    // Reciprocal function: calculates the reciprocal of each element in the tensor
    // Reciprocol is defined as: f(x) = 1 / x if abs(x) > epsilon, else 1 / epsilon
    static Tensor reciprocal(const Tensor& tensor, double epsilon = 1e-12) {
        Tensor absolute = abs(tensor);
        Tensor epsilonTensor = fill(tensor.shape(), epsilon, tensor.type());
        return select(greaterThan(absolute, epsilonTensor), 1 / tensor, 1 / epsilonTensor);
    }

    // Variance function: calculates the variance between tensors along the specified axes
    static Tensor variance(const Tensor& tensor, const Tensor& meanTensor, const std::vector<size_t>& axes) {
        Tensor diff = tensor - meanTensor;
        Tensor squaredDiff = diff * diff;
        Tensor summedSquaredDiff = sum(squaredDiff, axes);

        // Ensure floating-point division
        float totalElements = 1.0f;
        for (size_t axis : axes) {
            totalElements *= static_cast<float>(tensor.shape()[axis]);
        }

        return summedSquaredDiff / totalElements;
    }

    // Variance function: calculates the variance for the entire tensor (all elements)
    static Tensor variance(const Tensor& tensor, const Tensor& meanTensor) {
        Tensor diff = tensor - meanTensor; // Broadcasting is handled automatically
        Tensor squaredDiff = diff * diff;
        size_t totalElements = tensor.shape().size();
        return sum(squaredDiff) / static_cast<double>(totalElements);
    }

    // Transpose function: swaps two dimensions of the tensor
    static Tensor transpose(const Tensor& tensor, size_t dim0, size_t dim1) {
        auto shape = tensor.shape();
        auto rank = shape.rank();

        if (dim0 >= rank || dim1 >= rank) {
            throw std::invalid_argument("Invalid dimensions for transpose.");
        }

        // Create permutation vector for axes
        std::vector<size_t> perm(rank);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[dim0], perm[dim1]);

        // Transpose the tensor using the permutation
        return sdnn::transpose(tensor, perm);
    }

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

        return sum(a * b);
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
            throw std::invalid_argument("Matrix and vector dimensions must match for multiplication. Mismatch in shapes: " + shapeA.toString() + " and " + shapeB.toString());
        }

        Shape resultShape({shapeA[0]});
        Tensor result = zeros(resultShape, a.type());

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < shapeA[0]; ++i) {
            float sum = 0;
            for (int j = 0; j < shapeA[1]; ++j) {
                sum += a.at<float>(i * shapeA[1] + j) * b.at<float>(j);
            }
            result.set(i, sum);
        }

        return result;
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
            throw std::invalid_argument("Matrix dimensions must match for multiplication. Mismatch in shapes: " + shapeA.toString() + " and " + shapeB.toString());
        }

        Shape resultShape({shapeA[0], shapeB[1]});
        Tensor result = zeros(resultShape, a.type());

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < shapeA[0]; ++i) {
            for (int j = 0; j < shapeB[1]; ++j) {
                double sum = 0;
                for (int k = 0; k < shapeA[1]; ++k) {
                    sum += a.at<double>({static_cast<size_t>(i), static_cast<size_t>(k)}) * b.at<double>({static_cast<size_t>(k), static_cast<size_t>(j)});
                }
                result.set(i * shapeB[1] + j, sum);
            }
        }

        return result;
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
        
        Tensor result = zeros(Shape(resultShape), a.type());

        // Iterate over all batch dimensions
        std::vector<size_t> batchIndices(rank - 2, 0);
        do {
            #pragma omp parallel for collapse(3)
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
                    result.set(computeFlatIndex(Shape(resultShape), resultIndices), sum);
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

        #pragma omp parallel for collapse(3)
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

        Tensor result = zeros(Shape({batchSize, m, p}), a.type());

        #pragma omp parallel for collapse(3)
        for (int batch = 0; batch < batchSize; ++batch) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < p; ++j) {
                    double sum = 0;
                    for (int k = 0; k < n; ++k) {
                        double aVal = a.at<double>((batch * m * n) + (i * n) + k);
                        double bVal = b.at<double>(k * p + j);
                        sum += aVal * bVal;
                    }
                    result.set((batch * m * p) + (i * p) + j, sum);
                }
            }
        }

        return result;
    }

    template<typename ShapeType>
    static bool incrementIndices(std::vector<size_t>& indices, 
                                 const ShapeType& shape, 
                                 const std::vector<bool>* axesToSum = nullptr, 
                                 std::vector<int>* resultIndices = nullptr) {
        for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
            ++indices[i];
            if (indices[i] < static_cast<size_t>(getShapeElement(shape, i))) {
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
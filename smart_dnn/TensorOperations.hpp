#ifndef TENSOR_OPERATIONS_HPP
#define TENSOR_OPERATIONS_HPP

#include "Tensor.hpp"
#include <stack>

class TensorOperations {
    public:
        template<typename... Args>
        static Tensor ones(Args... args) {
            Shape dimensions{args...};
            return {dimensions, 1.0f};
        }

        static Tensor identity(int size) {
            Tensor result{{size, size}};
            for (int i = 0; i < size; ++i) {
                result.data[i * size + i] = 1.0f;
            }
            return result;
        }

    static void printTensor(const Tensor& tensor) {
        std::cout << tensor << std::endl;
    }

    static Tensor randomn(const Shape& shape) {
        Tensor result(shape);
        result.randomize(-3.0f, 3.0f);
        return result;
    }

    static Tensor transpose(Tensor tensor, int dim1, int dim2) {
        tensor.transpose(dim1, dim2);
        return tensor;
    }

    static int flattenIndex(const std::vector<int>& indices, const Shape& shape) {
        if (indices.size() != shape.rank()) {
            throw std::invalid_argument("Indices size must match the tensor rank");
        }
        int flatIndex = 0;
        int stride = 1;
        for (int i = shape.rank() - 1; i >= 0; --i) {
            flatIndex += indices[i] * stride;
            stride *= shape[i];
        }
        return flatIndex;
    }

    static std::vector<int> getIndices(int flatIndex, const Shape& shape) {
        if (flatIndex < 0 || flatIndex >= shape.size()) {
            throw std::out_of_range("Flat index out of range");
        }

        std::vector<int> indices(shape.rank(), 0);
        for (int i = shape.rank() - 1; i >= 0; --i) {
            indices[i] = flatIndex % shape[i];
            flatIndex /= shape[i];
        }
        return indices;
    }

    static std::vector<int> getBroadcastShape(const std::vector<int>& shape1, const std::vector<int>& shape2) {
        std::vector<int> resultShape;

        int rank1 = shape1.size();
        int rank2 = shape2.size();
        int maxRank = std::max(rank1, rank2);

        for (int i = 0; i < maxRank; ++i) {
            int dim1 = (i < rank1) ? shape1[rank1 - 1 - i] : 1;
            int dim2 = (i < rank2) ? shape2[rank2 - 1 - i] : 1;

            if (dim1 == dim2 || dim1 == 1 || dim2 == 1) {
                resultShape.push_back(std::max(dim1, dim2));
            } else {
                throw std::invalid_argument("Tensor dimensions do not match for broadcasting");
            }
        }

        std::reverse(resultShape.begin(), resultShape.end());
        return resultShape;
    }

    static Tensor matmul(Tensor& a, Tensor& b) {
        int aRank = a.shape().rank();
        int bRank = b.shape().rank();

        if (aRank == 1 && bRank == 1) {
            return {{1}, matmul1D(a, b)};
        }

        if (aRank == 2 && bRank == 1) {
            return matmul1D2D(a, b);
        }

        if (aRank == 1 && bRank == 2) {
            TensorShapeRestorer restorer(a, {1, a.shape()[0]}); 
            return matmul2D(a, b);
        }

        if (aRank == 2 && bRank == 2) {
            return matmul2D(a, b);
        }

        if (aRank == 1 && bRank > 2) {
            TensorShapeRestorer restorer(a, {1, a.shape()[0]}); 
            return matmulnD(a, b);
        }

        if (aRank > 2 && bRank == 1) {
            TensorShapeRestorer restorer(b, {1, b.shape()[0]}); 
            return matmulnD(a, b);
        }

        if (aRank > 2 && bRank > 2) {
            return matmulnD(a, b);
        }

        throw std::invalid_argument("Invalid tensor ranks for dot product.");
    }

    private:

        // Helper class to restore the original shape of a tensor
        class TensorShapeRestorer {
            public:
                TensorShapeRestorer(Tensor& tensor, const Shape& newShape) 
                    : tensorRef(tensor), originalShape(tensor.shape()) {
                        tensorRef.reshape(newShape);
                    }

                ~TensorShapeRestorer() {
                    tensorRef.reshape(originalShape);
                }

            private:
                Tensor& tensorRef;
                Shape originalShape;
        };

        static float matmul1D(const Tensor& a, const Tensor& b) {
            if (a.shape() != b.shape()) {
                throw std::invalid_argument("Tensors must have the same shape");
            }

            float result = 0;
            for (int i = 0; i < a.shape().size(); ++i) {
                result += a.data[i] * b.data[i];
            }
            return result;
        }

        static Tensor matmul1D2D(const Tensor& matrix, const Tensor& vector) {
            if (matrix.shape()[1] != vector.shape()[0]) {
                throw std::invalid_argument("Invalid dimensions for vector-matrix multiplication");
            }

            std::vector<float> resultData(matrix.shape()[0], 0.0f);
            const float* vectorData = vector.getData().data();
            const float* matrixData = matrix.getData().data();

            int numRows = matrix.shape()[0];
            int numCols = vector.shape()[0];

            for (int i = 0; i < numRows; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < numCols; ++j) {
                    sum += vectorData[j] * matrixData[i * numCols + j];
                }
                resultData[i] = sum;
            }

            return {{matrix.shape()[0]}, resultData};
        }

        static Tensor matmul2D(const Tensor& a, const Tensor& b) {
            if (a.shape()[1] != b.shape()[0]) {
                throw std::invalid_argument("Invalid dimensions for matrix-matrix multiplication: The columns of a must equal the rows of b, but got A:" +
                std::to_string(a.shape()[1]) + " and B:" + std::to_string(b.shape()[0]));
            }

            std::vector<int> resultShape = {a.shape()[0], b.shape()[1]};
            std::vector<float> resultData(resultShape[0] * resultShape[1], 0.0f);

            for (int i = 0; i < a.shape()[0]; ++i) {
                for (int j = 0; j < b.shape()[1]; ++j) {
                    for (int k = 0; k < a.shape()[1]; ++k) {
                        resultData[i * b.shape()[1] + j] += a({i,k}) * b({k,j});
                    }
                }
            }

            return {Shape(resultShape), resultData};
        }

        // Batched matrix multiplication
        static Tensor matmulnD(const Tensor& a, const Tensor& b) {
            if (a.shape().rank() < 2 || b.shape().rank() < 2) {
                throw std::invalid_argument("Tensors must have at least 2 dimensions for matrix multiplication.");
            }

            if (a.shape()[a.shape().rank() - 1] != b.shape()[b.shape().rank() - 2]) {
                throw std::invalid_argument("Inner dimensions must match for batched matrix multiplication.");
            }

            std::vector<int> aShape = a.shape().dimensions;
            std::vector<int> bShape = b.shape().dimensions;
            std::vector<int> batchShape = getBroadcastShape(std::vector<int>(aShape.begin(), aShape.end()- 2),
                                                            std::vector<int>(bShape.begin(), bShape.end()- 2));

            batchShape.push_back(a.shape()[a.shape().rank() - 2]);
            batchShape.push_back(b.shape()[b.shape().rank() - 1]);

            Tensor result(Shape(batchShape), std::vector<float>(Shape(batchShape).size(), 0.0f));

            for (int batch = 0; batch < result.shape().size() / (batchShape[batchShape.size() - 2] * batchShape[batchShape.size() - 1]); ++batch) {
                for (int i = 0; i < batchShape[batchShape.size() - 2]; ++i) {
                    for (int j = 0; j < batchShape[batchShape.size() - 1]; ++j) {
                        float sum = 0.0f;
                        for (int k = 0; k < a.shape()[a.shape().rank() - 1]; ++k) {
                            int aIndex = batch * a.shape()[a.shape().rank() - 2] * a.shape()[a.shape().rank() - 1] + i * a.shape()[a.shape().rank() - 1] + k;
                            int bIndex = batch * b.shape()[b.shape().rank() - 2] * b.shape()[b.shape().rank() - 1] + k * b.shape()[b.shape().rank() - 1] + j;
                            sum += a.getData()[aIndex] * b.getData()[bIndex];
                        }
                        int resultIndex = batch * batchShape[batchShape.size() - 2] * batchShape[batchShape.size() - 1] + i * batchShape[batchShape.size() - 1] + j;
                        result.getData()[resultIndex] = sum;
                    }
                }
            }

            return result;
        }



    TensorOperations() = delete;
};

#endif // TENSOR_OPERATIONS_HPP
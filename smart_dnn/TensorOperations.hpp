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
        std::cout << "Tensor Shape: " << tensor.shape() << std::endl;
        int nDims = tensor.shape().rank();
        std::vector<int> indices(nDims, 0);
        std::vector<float> data = tensor.getData();
        std::stack<int> bracketStack;

        for (int i = 0; i < data.size(); ++i) {
            std::cout << data[i] << " ";
            indices[nDims - 1] += 1;
            for (int j = nDims - 1; j >= 0; --j) {
                if (indices[j] == tensor.shape()[j]) {
                    indices[j] = 0;
                    if (j > 0) {
                        std::cout << std::endl;
                        indices[j - 1] += 1;
                    }
                }
            }
        }
        std::cout << std::endl;
    }

    static int flattenIndex(const std::vector<int>& indices, const Shape& shape) {
        int flatIndex = 0;
        int stride = 1;
        for (int i = shape.rank() - 1; i >= 0; --i) {
            flatIndex += indices[i] * stride;
            stride *= shape[i];
        }
        return flatIndex;
    }

    static std::vector<int> getIndices(int flatIndex, const Shape& shape) {
        std::vector<int> indices(shape.rank(), 0);
        for (int i = shape.rank() - 1; i >= 0; --i) {
            indices[i] = flatIndex % shape[i];
            flatIndex /= shape[i];
        }
        return indices;
    }

    static Tensor dotprod(const Tensor& a, const Tensor& b) {
        if (a.shape().rank() == 1 && b.shape().rank() == 1) {
            return {{1}, computeScalar(a, b)};
        } else if ((a.shape().rank() == 2 && b.shape().rank() == 1)) {
            return matrixVectProduct(a, b);
        } else {
            throw std::invalid_argument("Invalid dimensions for dot product");
        }

        return {};
    }

    private:

        static float computeScalar(const Tensor& a, const Tensor& b) {
            if (a.shape() != b.shape()) {
                throw std::invalid_argument("Tensors must have the same shape");
            }

            float result = 0;
            for (int i = 0; i < a.shape().size(); ++i) {
                result += a.data[i] * b.data[i];
            }
            return result;
        }

        static Tensor matrixVectProduct(const Tensor& matrix, const Tensor& vector) {
            if (matrix.shape()[1] != vector.shape()[0]) {
                throw std::invalid_argument("Invalid dimensions for vector-matrix multiplication");
            }

            std::vector<float> resultData(matrix.shape()[0], 0.0f);
            for (int i = 0; i < matrix.shape()[0]; ++i) {
                for (int j = 0; j < vector.shape().size(); ++j){
                    resultData[i] += vector.data[j] * matrix.data[i * vector.shape().size() + j];
                }
            }

            return {{matrix.shape()[0]}, resultData};
        }

    TensorOperations() = delete;
};

#endif // TENSOR_OPERATIONS_HPP
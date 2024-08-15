#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H

#include "Tensor.hpp"
#include <stack>

class TensorOperations {
    public:
        TensorOperations() = delete;

        template<typename... Args>
        static Tensor ones(Args... args) {
            Shape dimensions{args...};
            return Tensor(dimensions, 1.0f);
        }

        static Tensor identity(int size) {
            Tensor result{{size, size}};
            for (int i = 0; i < size; ++i) {
                result.data[i * size + i] = 1.0f;
            }
            return result;
        }

    static void printTensor(const Tensor& tensor) {
        std::cout << "Tensor: " << tensor._shape.toString() << std::endl;
        int nDims = tensor._shape.rank();
        std::vector<int> indices(nDims, 0);
        std::vector data = tensor.getData();
        std::stack<int> bracketStack;

        for (int i = 0; i < data.size(); ++i) {
            std::cout << data[i] << " ";
            indices[nDims - 1] += 1;
            for (int j = nDims - 1; j >= 0; --j) {
                if (indices[j] == tensor._shape.dimensions[j]) {
                    indices[j] = 0;
                    if (j > 0) {
                        std::cout << std::endl;
                        indices[j - 1] += 1;
                    }
                }
            }
        }
    }

    static int flattenIndex(const std::vector<int>& indices, const Shape& shape) {
        int flatIndex = 0;
        int stride = 1;
        for (int i = shape.rank() - 1; i >= 0; --i) {
            flatIndex += indices[i] * stride;
            stride *= shape.dimensions[i];
        }
        return flatIndex;
    }

    static std::vector<int> getIndices(int flatIndex, const Shape& shape) {
        std::vector<int> indices(shape.rank(), 0);
        for (int i = shape.rank() - 1; i >= 0; --i) {
            indices[i] = flatIndex % shape.dimensions[i];
            flatIndex /= shape.dimensions[i];
        }
        return indices;
    }
};

#endif // TENSOR_OPERATIONS_H
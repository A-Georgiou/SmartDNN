#ifndef TENSOR_HELPERS_HPP
#define TENSOR_HELPERS_HPP

#include <gtest/gtest.h>
#include "../smart_dnn/Tensor/Tensor.hpp"
#include "../smart_dnn/Tensor/TensorData.hpp"

namespace smart_dnn {

    template <typename T>
    inline void ValidateTensorShape(const Tensor<T>& tensor, int expectedRank, int expectedSize, const std::vector<int>& expectedDimensions) {
        ASSERT_EQ(tensor.getShape().rank(), expectedRank);
        ASSERT_EQ(tensor.getShape().size(), expectedSize);
        ASSERT_EQ(tensor.getShape().getDimensions().size(), expectedDimensions.size());  // Ensure both vectors are the same length
        for (size_t i = 0; i < expectedDimensions.size(); ++i) {
            ASSERT_EQ(tensor.getShape()[i], expectedDimensions[i]);
        }
    }

    template <typename T>
    inline void ValidateTensorData(const Tensor<T>& tensor, const std::vector<T>& expectedData) {
        ASSERT_EQ(tensor.getData().size(), expectedData.size());
        auto tensor_it = tensor.getData().data();
        auto expected_it = expectedData.begin();
        while (tensor_it != tensor.getData().data() + tensor.getData().size() && expected_it != expectedData.end()) {
            ASSERT_NEAR(*tensor_it, *expected_it, 1e-6);
            ++tensor_it;
            ++expected_it;
        }
    }

    template <typename T>
    inline void ValidateRandomTensor(const Tensor<T>& tensor, T min = 0.0, T max = 1.0) {
        for (T value : tensor.getData()) {
            ASSERT_GE(value, min);
            ASSERT_LE(value, max);
        }
    }

    template <typename T>
    inline bool TensorEquals(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
        bool equal = true;
        if (tensor1.getShape() != tensor2.getShape()) {
            return false;
        }
        for (size_t i = 0; i < tensor1.getData().size(); ++i) {
            if (std::abs(tensor1.getData()[i] - tensor2.getData()[i]) > 1e-6) {
                equal = false;
                break;
            }
        }
        return equal;
    }

} // namespace smart_dnn

#endif // TENSOR_HELPERS_HPP
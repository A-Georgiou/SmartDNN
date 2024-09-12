#ifndef TENSOR_HELPERS_HPP
#define TENSOR_HELPERS_HPP

#include <gtest/gtest.h>
#include <sstream>
#include "smart_dnn/tensor/Tensor.hpp"
#include "smart_dnn/tensor/TensorData.hpp"

namespace sdnn {

    template <typename T>
    inline void ValidateTensorShape(const Tensor<T>& tensor, int expectedRank, int expectedSize, const std::vector<int>& expectedDimensions) {
        EXPECT_EQ(tensor.getShape().rank(), expectedRank) 
            << "Expected rank: " << expectedRank << ", Actual rank: " << tensor.getShape().rank();
        EXPECT_EQ(tensor.getShape().size(), expectedSize)
            << "Expected size: " << expectedSize << ", Actual size: " << tensor.getShape().size();
        EXPECT_EQ(tensor.getShape().getDimensions().size(), expectedDimensions.size())
            << "Expected dimensions size: " << expectedDimensions.size() 
            << ", Actual dimensions size: " << tensor.getShape().getDimensions().size();
        
        for (size_t i = 0; i < expectedDimensions.size(); ++i) {
            EXPECT_EQ(tensor.getShape()[i], expectedDimensions[i])
                << "Mismatch at dimension " << i 
                << ". Expected: " << expectedDimensions[i] 
                << ", Actual: " << tensor.getShape()[i];
        }
    }

    template <typename T>
    inline void ValidateTensorData(const Tensor<T>& tensor, const std::vector<T>& expectedData) {
        EXPECT_EQ(tensor.getData().size(), expectedData.size())
            << "Expected data size: " << expectedData.size() 
            << ", Actual data size: " << tensor.getData().size();

        auto tensor_it = tensor.getData().data();
        auto expected_it = expectedData.begin();
        for (size_t i = 0; tensor_it != tensor.getData().data() + tensor.getData().size() && expected_it != expectedData.end(); ++i, ++tensor_it, ++expected_it) {
            EXPECT_NEAR(*tensor_it, *expected_it, 1e-6)
                << "Mismatch at index " << i 
                << ". Expected: " << *expected_it 
                << ", Actual: " << *tensor_it;
        }
    }

    template <typename T>
    inline void ValidateRandomTensor(const Tensor<T>& tensor, T min = 0.0, T max = 1.0) {
        for (size_t i = 0; i < tensor.getData().size(); ++i) {
            T value = tensor.getData()[i];
            EXPECT_GE(value, min) << "Value at index " << i << " is less than minimum. Value: " << value << ", Min: " << min;
            EXPECT_LE(value, max) << "Value at index " << i << " is greater than maximum. Value: " << value << ", Max: " << max;
        }
    }

    template <typename T>
    inline testing::AssertionResult TensorEquals(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
        if (tensor1.getShape() != tensor2.getShape()) {
            return testing::AssertionFailure() << "Tensor shapes do not match. "
                << "Shape1: " << tensor1.getShape().toString() 
                << ", Shape2: " << tensor2.getShape().toString();
        }

        for (size_t i = 0; i < tensor1.getData().size(); ++i) {
            if (std::abs(tensor1.getData()[i] - tensor2.getData()[i]) > 1e-6) {
                return testing::AssertionFailure() << "Tensor data mismatch at index " << i 
                    << ". Value1: " << tensor1.getData()[i] 
                    << ", Value2: " << tensor2.getData()[i];
            }
        }

        return testing::AssertionSuccess();
    }

} // namespace sdnn

#endif // TENSOR_HELPERS_HPP
#ifndef TENSOR_HELPERS_HPP
#define TENSOR_HELPERS_HPP

#include <gtest/gtest.h>
#include <sstream>
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/TensorAdapterBase.hpp"
#include "smart_dnn/tensor/TensorCreationUtil.hpp"

namespace sdnn {

    inline void ValidateTensorShape(const Tensor& tensor, int expectedRank, int expectedSize, const std::vector<int>& expectedDimensions) {
        EXPECT_EQ(tensor.shape().rank(), expectedRank) 
            << "Expected rank: " << expectedRank << ", Actual rank: " << tensor.shape().rank();
        EXPECT_EQ(tensor.shape().size(), expectedSize)
            << "Expected size: " << expectedSize << ", Actual size: " << tensor.shape().size();
        EXPECT_EQ(tensor.shape().getDimensions().size(), expectedDimensions.size())
            << "Expected dimensions size: " << expectedDimensions.size() 
            << ", Actual dimensions size: " << tensor.shape().getDimensions().size();
        
        for (size_t i = 0; i < expectedDimensions.size(); ++i) {
            EXPECT_EQ(tensor.shape()[i], expectedDimensions[i])
                << "Mismatch at dimension " << i 
                << ". Expected: " << expectedDimensions[i] 
                << ", Actual: " << tensor.shape()[i];
        }
    }

    template <typename T>
    inline void ValidateTensorData(const Tensor& tensor, const std::vector<T>& expectedData) {
        EXPECT_EQ(tensor.shape().size(), expectedData.size())
            << "Expected data size: " << expectedData.size() 
            << ", Actual data size: " << tensor.shape().size();

        for (size_t i = 0; i < tensor.shape().size(); ++i) {
            EXPECT_NEAR(tensor.at<T>(i), expectedData[i], 1e-6)
                << "Mismatch at index " << i 
                << ". Expected: " << expectedData[i] 
                << ", Actual: " << tensor.at<T>(i);
        }
    }


    inline testing::AssertionResult TensorEquals(const Tensor& tensor1, const Tensor& tensor2) {
        if (tensor1.type() != tensor2.type()) {
            return testing::AssertionFailure() << "Tensor types do not match.";
        }

        if (tensor1.shape() != tensor2.shape()) {
            return testing::AssertionFailure() << "Tensor shapes do not match. "
                << "Shape1: " << tensor1.shape().toString() 
                << ", Shape2: " << tensor2.shape().toString();
        }

        for (size_t i = 0; i < tensor1.shape().size(); ++i) {
            if (std::abs(tensor1.at<float>(i) - tensor2.at<float>(i)) > 1e-6) {
                return testing::AssertionFailure() << "Tensor data mismatch at index " << i 
                    << ". Value1: " << tensor1.at<float>(i) 
                    << ", Value2: " << tensor2.at<float>(i);
            }
        }

        return testing::AssertionSuccess();
    }

} // namespace sdnn

#endif // TENSOR_HELPERS_HPP
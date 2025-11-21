#ifndef TENSOR_HELPERS_HPP
#define TENSOR_HELPERS_HPP

#include <gtest/gtest.h>
#include "../smart_dnn/Tensor.hpp"

inline void ValidateTensorShape(const Tensor& tensor, int rank, int size, const std::vector<int>& dimensions) {
    ASSERT_EQ(tensor.shape().rank(), rank);
    ASSERT_EQ(tensor.shape().size(), size);
    for (int i = 0; i < rank; ++i) {
        ASSERT_EQ(tensor.shape()[i], dimensions[i]);
    }
}

inline void ValidateTensorData(const Tensor& tensor, const std::vector<float>& expectedData) {
    const float* data = tensor.getData();
    // Verify that tensor size matches expected data size
    ASSERT_EQ(static_cast<size_t>(tensor.shape().size()), expectedData.size()) 
        << "Tensor size does not match expected data size";
    for (size_t i = 0; i < expectedData.size(); ++i) {
        ASSERT_NEAR(data[i], expectedData[i], 1e-6);
    }
}

#endif // TENSOR_HELPERS_HPP
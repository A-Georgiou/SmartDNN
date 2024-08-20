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
    ASSERT_EQ(tensor.getData().size(), expectedData.size());
    for (size_t i = 0; i < expectedData.size(); ++i) {
        ASSERT_NEAR(tensor.getData()[i], expectedData[i], 1e-6);
    }
}

#endif // TENSOR_HELPERS_HPP
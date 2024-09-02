#ifndef TENSOR_HELPERS_HPP
#define TENSOR_HELPERS_HPP

#include <gtest/gtest.h>
#include "../smart_dnn/Tensor/Tensor.hpp"

namespace smart_dnn {

    template <typename T>
    inline void ValidateTensorShape(const Tensor<T>& tensor, int rank, int size, const std::vector<int>& dimensions) {
        ASSERT_EQ(tensor.getShape().rank(), rank);
        ASSERT_EQ(tensor.getShape().size(), size);
        for (int i = 0; i < rank; ++i) {
            ASSERT_EQ(tensor.getShape()[i], dimensions[i]);
        }
    }

    template <typename T>
    inline void ValidateTensorData(const Tensor<T>& tensor, const std::vector<T>& expectedData) {
        ASSERT_EQ(tensor.getData().size(), expectedData.size());
        auto tensor_it = tensor.getData().begin();
        auto expected_it = expectedData.begin();
        while (tensor_it != tensor.getData().end() && expected_it != expectedData.end()) {
            ASSERT_NEAR(*tensor_it, *expected_it, 1e-6);
            ++tensor_it;
            ++expected_it;
        }
    }

} // namespace smart_dnn

#endif // TENSOR_HELPERS_HPP
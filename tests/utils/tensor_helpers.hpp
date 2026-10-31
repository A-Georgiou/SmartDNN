#ifndef TENSOR_HELPERS_HPP
#define TENSOR_HELPERS_HPP

#include <gtest/gtest.h>
#include <cmath>
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

    // Expects zero-mean weights with the He standard deviation sqrt(2 / fanIn).
    template <typename T>
    inline void ExpectHeNormal(const Tensor<T>& weights, int fanIn) {
        double sum = 0.0, sumSq = 0.0;
        for (T value : weights.getData()) {
            sum += value;
            sumSq += static_cast<double>(value) * value;
        }
        const double n = static_cast<double>(weights.getData().size());
        const double mean = sum / n, stddev = std::sqrt(sumSq / n - mean * mean), expected = std::sqrt(2.0 / fanIn);
        EXPECT_NEAR(mean, 0.0, 0.1 * expected);
        EXPECT_NEAR(stddev, expected, 0.1 * expected);
    }

} // namespace smart_dnn

#endif // TENSOR_HELPERS_HPP
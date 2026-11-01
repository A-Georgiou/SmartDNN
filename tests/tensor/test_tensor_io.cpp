#ifndef TEST_TENSOR_IO_CPP
#define TEST_TENSOR_IO_CPP

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include "../../smart_dnn/Tensor/TensorIO.hpp"

namespace smart_dnn {

TEST(TensorIOTest, Int8StorageKeepsEachValueWithinHalfAStepOfItsBlock) {
    // A block of 32 small values followed by a block of 8 large ones, so each block needs its own scale.
    Tensor<float> original(Shape({5, 8}), 0.0f);
    for (size_t i = 0; i < 40; ++i) {
        original[i] = (i < 32 ? 0.01f : 10.0f) * std::sin(1.3f * static_cast<float>(i) + 0.2f);
    }
    original[3] = 0.0f;
    std::stringstream stream;
    int8TensorStorage(stream) = 1;

    saveTensor(stream, original);
    Tensor<float> restored(Shape({5, 8}), 1.0f);
    loadTensor(stream, restored);

    EXPECT_EQ(stream.str().size(), sizeof(uint32_t) + 2 * sizeof(int) + 2 * sizeof(float) + 40);
    for (size_t start : {size_t(0), size_t(32)}) {
        float largest = 0.0f;
        for (size_t i = start; i < std::min<size_t>(start + 32, 40); ++i) {
            largest = std::max(largest, std::abs(original[i]));
        }
        for (size_t i = start; i < std::min<size_t>(start + 32, 40); ++i) {
            EXPECT_LE(std::abs(restored[i] - original[i]), largest / 254.0f * 1.001f) << "value " << i;
        }
    }
    EXPECT_EQ(restored[3], 0.0f);
}

TEST(TensorIOTest, Int8StorageRejectsNonFiniteValues) {
    std::stringstream stream;
    int8TensorStorage(stream) = 1;

    EXPECT_THROW(saveTensor(stream, Tensor<float>({2}, {1.0f, std::nanf("")})), std::runtime_error);
}

} // namespace smart_dnn

#endif // TEST_TENSOR_IO_CPP

#include <gtest/gtest.h>
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/tensor/Backend/Default/AdvancedTensorOperations.hpp"

namespace sdnn {

class TensorOperationsTest : public ::testing::Test {
protected:
    Tensor createTensor(const std::vector<int>& shape_, const std::vector<float>& values) {
        return Tensor(Shape(shape_), values);
    }
};


TEST_F(TensorOperationsTest, ReciprocalTest) {
    Tensor a = createTensor({2, 2}, {0.5f, 0.25f, 0.125f, 0.0625f});
    Tensor b = createTensor({2, 2}, {1.0f, 2.0f, 4.0f, 8.0f});

    Tensor result = AdvancedTensorOperations::reciprocal(a);
    Tensor result2 = AdvancedTensorOperations::reciprocal(b);

    std::vector<float> expected = {2.0f, 4.0f, 8.0f, 16.0f};
    std::vector<float> expected2 = {1.0f, 0.5f, 0.25f, 0.125f};
    for (size_t i = 0; i < result.shape().size(); ++i) {
        EXPECT_NEAR(result.at<float>(i), expected[i], 1e-5);
        EXPECT_NEAR(result2.at<float>(i), expected2[i], 1e-5);
    }
}

}
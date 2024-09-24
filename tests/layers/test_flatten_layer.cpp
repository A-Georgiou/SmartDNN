#ifndef TEST_FLATTEN_LAYER_CPP
#define TEST_FLATTEN_LAYER_CPP

#include "gtest/gtest.h"
#include "smart_dnn/layers/FlattenLayer.hpp"
#include "smart_dnn/Tensor/TensorBase.hpp"

namespace sdnn {

    TEST(FlattenLayer, Forward) {
        FlattenLayer layer;
        Tensor input = Tensor({2, 3, 4});
        Tensor output = layer.forward(input);
        EXPECT_EQ(output.shape(), Shape({2, 3 * 4}));
    }

    TEST(FlattenLayer, Backward) {
        FlattenLayer layer;
        Tensor input = Tensor({2, 3, 4});
        Tensor output = layer.forward(input);
        Tensor gradOutput = Tensor({2, 3 * 4});
        Tensor gradInput = layer.backward(gradOutput);
        EXPECT_EQ(gradInput.shape(), input.shape());
    }

}



#endif // TEST_FLATTEN_LAYER_CPP
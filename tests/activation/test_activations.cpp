#ifndef TEST_ACTIVATIONS_CPP
#define TEST_ACTIVATIONS_CPP

#include "../utils/tensor_helpers.hpp"
#include "../smart_dnn/Activations/ReLU.hpp"


/*
    
    VALID ACTIVATION TESTS

*/

// ReLU should simply remove all negative values from the input tensor
// Returning an output range from 0 -> inf 
TEST(TensorActivationsValidTest, ValidReLUForward) {
    ReLU relu{};
    Tensor input({2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    Tensor output = relu.forward(input);

    ValidateTensorShape(output, 2, 6, {2, 3});
    ValidateTensorData(output, {1.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f});
}

// Back propagation of ReLU where we have f(x <= 0) = 0 and f(x > 0) = x
// should result in f'(x) = 1 for x > 0 and 0 for x <= 0
TEST(TensorActivationsValidTest, ValidReLUBackward) {
    ReLU relu{};
    Tensor input({2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    Tensor output = relu.forward(input);
    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gradInput = relu.backward(input, gradOutput);

    ValidateTensorShape(gradInput, 2, 6, {2, 3});
    ValidateTensorData(gradInput, {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
}

#endif // TEST_ACTIVATIONS_CPP
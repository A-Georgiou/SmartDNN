#ifndef TEST_ACTIVATIONS_CPP
#define TEST_ACTIVATIONS_CPP

#include "../utils/tensor_helpers.hpp"
#include "../smart_dnn/Activations/ReLU.hpp"
#include "../smart_dnn/Activations/LeakyReLU.hpp"
#include "../smart_dnn/Activations/Sigmoid.hpp"
#include "../smart_dnn/Activations/Softmax.hpp"
#include "../smart_dnn/Activations/Tanh.hpp"


/*
    
    VALID ACTIVATION TESTS

*/

// ReLU TESTS

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

// LeakyReLU TESTS

// LeakyReLU should fix the problem with vanishing gradients caused by ReLU
// by allowing a small gradient for x <= 0
// Defined by f(x) = x for x > 0 and f(x) = alpha * x for x <= 0 where alpha is initialised to 0.01f
TEST(TensorActivationsValidTest, ValidLeakyReLUForward) {
    LeakyReLU leakyReLU{};
    Tensor input({2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    Tensor output = leakyReLU.forward(input);

    ValidateTensorShape(output, 2, 6, {2, 3});
    ValidateTensorData(output, {1.0f, -0.02f, 3.0f, -0.04f, 5.0f, -0.06f});
}

// Back propagation of LeakyReLU where we have f(x <= 0) = alpha * x and f(x > 0) = x
// should result in f'(x) = 1 for x > 0 and f'(x <= 0) = alpha * 1
TEST(TensorActivationsValidTest, ValidLeakyReLUBackward) {
    LeakyReLU leakyReLU{};
    Tensor input({2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    Tensor output = leakyReLU.forward(input);
    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gradInput = leakyReLU.backward(input, gradOutput);

    ValidateTensorShape(gradInput, 2, 6, {2, 3});
    ValidateTensorData(gradInput, {1.0f, 0.01, 1.0f, 0.01f, 1.0f, 0.01f});
}

// Sigmoid Tests

// Sigmoid should squash the input tensor values to a range of 0 -> 1
// Defined by f(x) = 1 / (1 + exp(-x))
TEST(TensorActivationsValidTest, ValidSigmoidForward) {
    Sigmoid sigmoid{};
    Tensor input({2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    Tensor output = sigmoid.forward(input);

    ValidateTensorShape(output, 2, 6, {2, 3});
    ValidateTensorData(output, {0.731058f, 0.119202f, 0.952574f, 0.017986f, 0.9933071f, 0.0024726f});
}

// Back propagation of Sigmoid where we have f(x) = 1 / (1 + exp(-x))
// should result in f'(x) = f(x) * (1 - f(x))
TEST(TensorActivationsValidTest, ValidSigmoidBackward) {
    Sigmoid sigmoid{};
    Tensor input({2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    Tensor output = sigmoid.forward(input);
    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gradInput = sigmoid.backward(input, gradOutput);

    ValidateTensorShape(gradInput, 2, 6, {2, 3});
    ValidateTensorData(gradInput, {0.196612f, 0.104994f, 0.045177f, 0.017663f, 0.006648f, 0.0024665f});
}

// Softmax Tests

// Softmax should squash the input tensor values to a range of 0 -> 1
// Defined by f(x) = exp(x - max(x)) / sum(exp(x - max(x)))
TEST(TensorActivationsValidTest, ValidSoftmaxForward) {
    Softmax softmax{};
    Tensor input({2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    Tensor output = softmax.forward(input);

    ValidateTensorShape(output, 2, 6, {2, 3});
    ValidateTensorData(output, {0.01586178f, 0.0007897114f, 0.1172036f, 0.0001069f, 0.8660237f, 0.000014f});
}

// Back propagation of Softmax where we have f(x) = exp(x - max(x)) / sum(exp(x - max(x)))
// should result in f'(x) = f(x) * (1 - f(x))
// Grad Output is very small, using scientific notation in this case.
TEST(TensorActivationsValidTest, ValidSoftmaxBackward){
    Softmax softmax{};
    Tensor input({2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    Tensor output = softmax.forward(input);
    Tensor gradOutput({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor gradInput = softmax.backward(input, gradOutput);
    
    TensorOperations::printTensor(gradInput);

    ValidateTensorShape(gradInput, 2, 6, {2, 3});
    ValidateTensorData(gradInput, {
        -3.20441e-10, -7.19433e-11, -3.95823e-09, -3.35587e-12, -3.9051e-08, 0.0f
    });
}

#endif // TEST_ACTIVATIONS_CPP
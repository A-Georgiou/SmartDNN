#ifndef LEAKY_RELU_HPP
#define LEAKY_RELU_HPP

#include "smart_dnn/Activation.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include <algorithm>

namespace sdnn {

/*

    Leaky ReLU Activation Function
    ------------------------

    f(x) = x if x > 0, alpha * x otherwise
    f'(x) = 1 if x > 0, alpha otherwise

*/

class LeakyReLU : public Activation {
public:
    explicit LeakyReLU(float alpha = 0.01f) : alpha(alpha) {}

    Tensor forward(const Tensor& input) const override {
        return selectMax(input, alpha * input);
    }

   Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        Tensor input_ones = ones(input.shape(), input.type());
        Tensor input_alpha = fill(input.shape(), alpha, input.type());
        return select(greaterThan(input, 0), input_ones, input_alpha) * gradOutput;
    }

private:
    float alpha;
};

} // namespace sdnn

#endif // LEAKY_RELU_HPP

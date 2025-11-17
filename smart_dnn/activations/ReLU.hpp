#ifndef RELU_HPP
#define RELU_HPP

#include "smart_dnn/Activation.hpp"

namespace sdnn {

/*

    ReLU Activation Function
    ------------------------

    f(x) = max(0, x)
    f'(x) = 1 if x > 0, 0 otherwise

*/

class ReLU : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        return selectMax(input, 0);
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        return greaterThan(input, 0) * gradOutput;
    }
};

} // namespace sdnn

#endif // RELU_HPP
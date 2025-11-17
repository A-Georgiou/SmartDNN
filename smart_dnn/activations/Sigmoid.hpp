#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "smart_dnn/Activation.hpp"

namespace sdnn {

/*

    Sigmoid Activation Function
    ------------------------

    f(x) = 1 / (1 + exp(-x))
    f'(x) = f(x) * (1 - f(x))

*/

class Sigmoid : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        return 1 / (1 + exp(input * -1));
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        Tensor sig = forward(input);
        return sig * (1 - sig) * gradOutput;
    }
};

} // namespace sdnn

#endif // SIGMOID_HPP
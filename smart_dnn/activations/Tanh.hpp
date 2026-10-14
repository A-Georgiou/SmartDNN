#ifndef TANH_HPP
#define TANH_HPP

#include <cmath>
#include "smart_dnn/Activation.hpp"

namespace sdnn {

/*

    Tanh Activation Function
    ------------------------

    f(x) = (exp(2x) - 1) / (exp(2x) + 1)
    f'(x) = 1 - tanh(x)^2

*/

class Tanh : public Activation {
public:
    Tensor forward(const Tensor& input) const override {
        return tanh(input);
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        Tensor tanh_input = tanh(input);
        return (1 - tanh_input * tanh_input) * gradOutput;
    }
};

} // namespace sdnn

#endif // TANH_HPP
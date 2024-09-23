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
        return apply(input, [](auto& x) { x = (std::exp(x * 2) - 1) / (std::exp(x * 2) + 1); });
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        return apply(input, [](auto& x) {  x = 1 - std::tanh(x) * std::tanh(x);  }) * gradOutput;
    }
};

} // namespace sdnn

#endif // TANH_HPP
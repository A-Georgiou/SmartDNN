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
        return apply(input, [](auto& x) { x = std::max(0.0, x); });
    }

    Tensor backward(const Tensor& input, const Tensor& gradOutput) const override {
        return apply(input, [](auto& x) { x = x > 0 ? 1 : 0; }) * gradOutput;
    }
};

} // namespace sdnn

#endif // RELU_HPP
#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "smart_dnn/tensor/TensorBase.hpp"

namespace sdnn {

class Activation {
public:
    virtual ~Activation() = default;
    virtual Tensor forward(const Tensor& input) const = 0;
    virtual Tensor backward(const Tensor& input, const Tensor& gradOutput) const= 0;
};

} // namespace sdnn

#endif // ACTIVATION_HPP
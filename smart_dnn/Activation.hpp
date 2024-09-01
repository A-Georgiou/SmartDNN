#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "Tensor/Tensor.hpp"
#include "Tensor/TensorConfig.hpp"

namespace smart_dnn {

class Activation {
public:
    virtual ~Activation() = default;
    virtual ConfiguredTensor<> forward(const ConfiguredTensor<>& input) const = 0;
    virtual ConfiguredTensor<> backward(const ConfiguredTensor<>& input, const ConfiguredTensor<>& gradOutput) const= 0;
};

} // namespace smart_dnn

#endif // ACTIVATION_HPP
#include "Tensor.hpp"

class Activation {
public:
    virtual ~Activation() = default;
    virtual Tensor forward(const Tensor& input) const = 0;
    virtual Tensor backward(const Tensor& input, const Tensor& gradOutput) const= 0;
};
#ifndef LAYER_HPP
#define LAYER_HPP

#include "Tensor.hpp"
#include "Optimizer.hpp"

class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor forward(Tensor& input) = 0;
    virtual Tensor backward(Tensor& gradOutput) = 0;
    virtual void updateWeights(Optimizer& optimizer) = 0;

protected:
    Tensor weights;
    Tensor weightGradients;
    Tensor biases;
    Tensor biasGradients;
    Tensor input;
};

#endif // LAYER_HPP

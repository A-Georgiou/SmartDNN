#ifndef LAYER_H
#define LAYER_H

#include "Tensor.hpp"
#include "Optimizer.hpp"

class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradOutput) = 0;
    virtual void updateWeights(Optimizer& optimizer) = 0;

protected:
    Tensor weights;
    Tensor weightGradients;
    Tensor biases;
    Tensor biasGradients;
};

#endif // LAYER_H
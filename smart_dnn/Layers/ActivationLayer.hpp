#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "../Layer.hpp"
#include "../Activation.hpp"

class ActivationLayer : public Layer {
public:
    ActivationLayer(Activation* activation) : activation(activation) {}

    ~ActivationLayer() {
        delete activation;
    }

    Tensor forward(Tensor& input) override {
        this->input = input;
        return activation->forward(input);
    }

    Tensor backward(Tensor& gradOutput) override {
        return activation->backward(input, gradOutput);
    }

    void updateWeights(Optimizer& optimizer) override {
        // No weights to update in an activation layer.
    }

private:
    Activation* activation;
    Tensor input;
};

#endif // ACTIVATION_LAYER_HPP
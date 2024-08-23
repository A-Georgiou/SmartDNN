#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "../Layer.hpp"
#include "../Activation.hpp"
#include <optional>

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
        if (!input.has_value()) {
            throw std::runtime_error("Backward called before forward");
        }
        return activation->backward(input.value(), gradOutput);
    }

    void updateWeights(Optimizer& optimizer) override {
        // No weights to update in an activation layer.
    }

private:
    Activation* activation;
    std::optional<Tensor> input;
};

#endif // ACTIVATION_LAYER_HPP
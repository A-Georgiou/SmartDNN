#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Activation.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include <optional>

namespace sdnn {

class ActivationLayer : public Layer {
public:
    ActivationLayer(Activation* activation) : activation(activation) {}

    ~ActivationLayer() {
        delete activation;
    }

    Tensor forward(const Tensor& input) override {
        this->input = input;
        return activation->forward(input);
    }

    Tensor backward(const Tensor& gradOutput) override {
        return activation->backward((*input), gradOutput);
    }

private:
    Activation* activation;
    std::optional<Tensor> input;
};

} // namespace sdnn

#endif // ACTIVATION_LAYER_HPP
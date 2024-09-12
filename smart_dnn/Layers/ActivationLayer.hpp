#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Activation.hpp"
#include "smart_dnn/tensor/Tensor.hpp"
#include <optional>

namespace sdnn {

template <typename T=float>
class ActivationLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    ActivationLayer(Activation<T>* activation) : activation(activation) {}

    ~ActivationLayer() {
        delete activation;
    }

    TensorType forward(const TensorType& input) override {
        this->input = input;
        return activation->forward(input);
    }

    TensorType backward(const TensorType& gradOutput) override {
        return activation->backward((*input), gradOutput);
    }

private:
    Activation<T>* activation;
    std::optional<TensorType> input;
};

} // namespace sdnn

#endif // ACTIVATION_LAYER_HPP
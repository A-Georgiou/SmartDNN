#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Activation.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include <optional>

namespace smart_dnn {

template <typename T=float>
class ActivationLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    template <typename ActivationType>
    ActivationLayer(ActivationType&& activation) 
        : activation(std::make_unique<std::decay_t<ActivationType>>(std::forward<ActivationType>(activation))) {}

    TensorType forward(const TensorType& input) override {
        this->input = input;
        return activation->forward(input);
    }

    TensorType backward(const TensorType& gradOutput) override {
        return activation->backward((*input), gradOutput);
    }

private:
    std::unique_ptr<Activation<T>> activation;
    std::optional<TensorType> input;
};

} // namespace smart_dnn

#endif // ACTIVATION_LAYER_HPP
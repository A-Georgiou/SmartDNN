#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

#include "../Layer.hpp"
#include "../Activation.hpp"
#include <optional>
#include "../Tensor/Tensor.hpp"

namespace smart_dnn {

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
        return activation->backward(input.value(), gradOutput);
    }

    void updateWeights(Optimizer<T>& optimizer) override {
        // No weights to update in an activation layer.
    }

private:
    Activation<T>* activation;
    std::optional<TensorType> input;
};

} // namespace smart_dnn

#endif // ACTIVATION_LAYER_HPP
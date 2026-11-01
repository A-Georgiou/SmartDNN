#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Layer.hpp"
#include "smart_dnn/RandomEngine.hpp"

namespace smart_dnn {

template <typename T=float>
class DropoutLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    DropoutLayer(T dropoutRate) : dropoutRate(dropoutRate) {}

    TensorType forward(const TensorType& input) override {
        if (this->trainingMode) {
            // Inverted dropout: keep with probability 1 - rate and scale kept values by 1 / (1 - rate).
            const T keptScale = T(1) / (T(1) - dropoutRate);
            mask = TensorType(input.getShape());
            for (T& value : (*mask).getData()) {
                value = RandomEngine::getRand() > dropoutRate ? keptScale : T(0);
            }
            return input * (*mask);
        } else {
            return input; // No dropout during inference
        }
    }

    TensorType backward(const TensorType& gradOutput) override {
        if (gradOutput.getShape() != (*mask).getShape()) {
            throw std::invalid_argument("Mask not initialised or has wrong shape in Dropout Layer.");
        }
        if (this->trainingMode) {
            return gradOutput * (*mask);
        }
        return gradOutput;
    }

private:
    T dropoutRate;
    std::optional<TensorType> mask;
};

} // namespace smart_dnn

#endif // DROPOUT_LAYER_HPP
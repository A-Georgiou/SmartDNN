#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/Layer.hpp"

namespace sdnn {

class DropoutLayer : public Layer {
public:
    DropoutLayer(float dropoutRate) : dropoutRate(dropoutRate) {
        if (dropoutRate < 0 || dropoutRate >= 1) {
            throw std::invalid_argument("Dropout rate must be between 0 and 1.");
        }
    }

    Tensor forward(const Tensor& input) override {
        if (this->trainingMode) {
            mask = uniformRand(input.shape(), input.type());
            (*mask).apply([this](auto& x) { x = (x > dropoutRate ? 1 / (1 - dropoutRate) : 0); });
            return input * (*mask);
        } else {
            return input; // No dropout during inference
        }
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (this->trainingMode) {
            if (!mask) {
                throw std::runtime_error("Backward called before forward in Dropout Layer.");
            }
            if (gradOutput.shape() != mask->shape()) {
                throw std::invalid_argument("Gradient output shape mismatch in Dropout Layer.");
            }

            return gradOutput * (*mask);
        }
        return gradOutput;
    }

    void setTrainingMode(bool trainingMode) override {
        this->trainingMode = trainingMode;
    }

private:
    float dropoutRate;
    std::optional<Tensor> mask;
    bool trainingMode = true;
};

} // namespace sdnn

#endif // DROPOUT_LAYER_HPP
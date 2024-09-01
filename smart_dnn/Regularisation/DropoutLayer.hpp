#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "../Tensor/Tensor.hpp"
#include "../Tensor/TensorConfig.hpp"
#include "../Layer.hpp"
#include "../TensorOperations.hpp"

namespace smart_dnn {

class DropoutLayer : public Layer {
public:
    DropoutLayer(float dropoutRate) : dropoutRate(dropoutRate) {}

    ConfiguredTensor<> forward(ConfiguredTensor<>& input) override {
        if (trainingMode) {
            float rate = dropoutRate;
            mask = TensorOperations::randomn(input.getShape());
            mask = mask.apply([rate](float x) { return x > rate ? 1.0f : 0.0f; });
            return input * mask * (1.0f / (1.0f - dropoutRate));
        } else {
            return input; // No dropout during inference
        }
    }

    ConfiguredTensor<> backward(ConfiguredTensor<>& gradOutput) override {
        if (gradOutput.getShape() != mask.getShape()) {
            throw std::invalid_argument("Mask not initialised or has wrong shape in Dropout Layer.");
        }
        if (trainingMode) {
            return gradOutput * mask;
        }
        return gradOutput;
    }

    void updateWeights(Optimizer& optimizer) override {}

private:
    float dropoutRate;
    ConfiguredTensor<> mask;
};

} // namespace smart_dnn

#endif // DROPOUT_LAYER_HPP
#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "../Tensor.hpp"
#include "../Layer.hpp"
#include "../TensorOperations.hpp"

class DropoutLayer : public Layer {
public:
    DropoutLayer(float dropoutRate) : dropoutRate(dropoutRate) {}

    Tensor forward(Tensor& input) override {
        if (trainingMode) {
            float rate = dropoutRate;
            mask = TensorOperations::randomn(input.shape());
            mask = mask.apply([rate](float x) { return x > rate ? 1.0f : 0.0f; });
            return input * mask * (1.0f / (1.0f - dropoutRate));
        } else {
            return input; // No dropout during inference
        }
    }

    Tensor backward(Tensor& gradOutput) override {
        if (gradOutput.shape() != mask.shape()) {
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
    Tensor mask;
};


#endif // DROPOUT_LAYER_HPP
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
            Tensor mask = TensorOperations::randomn(input.shape());
            mask = mask.apply([this](float x) { return x > dropoutRate ? 1.0f : 0.0f; });
            return input * mask * (1.0f / (1.0f - dropoutRate));
        } else {
            return input; // No dropout during inference
        }
    }

    Tensor backward(Tensor& gradOutput) override {
        if (trainingMode) {
            return gradOutput * mask;
        }
        return gradOutput;
    }

private:
    float dropoutRate;
    Tensor mask;
};


#endif // DROPOUT_LAYER_HPP
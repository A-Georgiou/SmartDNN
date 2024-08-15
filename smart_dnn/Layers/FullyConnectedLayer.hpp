#include "../Tensor.hpp"

class FullyConnectedLayer {
public:
    FullyConnectedLayer(int inputSize, int outputSize) : weights({inputSize, outputSize}), biases({1, outputSize}) {
        weights.randomize(-1.0, 1.0);
        biases.fill(0.0);
    }

    Tensor forward(const Tensor& input) {
    }

    Tensor backward(const Tensor& gradOutput) {
    }

    void updateWeights(Optimizer& optimizer) {
    }

private:
    Tensor weights;
    Tensor biases;
    Tensor input;
    Tensor weightGradients;
    Tensor biasGradients;
};
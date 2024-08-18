#include "../Tensor.hpp"
#include "../Optimizer.hpp"
#include "../TensorOperations.hpp"
#include <vector>

class FullyConnectedLayer {
public:
    FullyConnectedLayer(int inputSize, int outputSize) : weights({inputSize, outputSize}), biases({1, outputSize}) {
        weights.randomize(-1.0, 1.0);
        biases.fill(0.0);
    }

    Tensor forward(Tensor& input) {
        this->input = input;
        Tensor output = TensorOperations::matmul(input, weights);  
        output.add(this->biases);
        return output;
    }

    Tensor backward(const Tensor& gradOutput) {
        
    }

    void updateWeights(Optimizer& optimizer) {
        std::vector<Tensor> weights = {this->weights, this->biases};
        std::vector<Tensor> weightGradients = {this->weightGradients, this->biasGradients};
        optimizer.optimize(weights, weightGradients, 0.01f);
    }

private:
    Tensor weights;
    Tensor biases;
    Tensor input;
    Tensor weightGradients;
    Tensor biasGradients;
};
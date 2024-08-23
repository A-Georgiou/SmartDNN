#ifndef BATCHNORM_LAYER_HPP
#define BATCHNORM_LAYER_HPP

#include "Layer.hpp"
#include "../Tensor.hpp"
#include "../TensorOperations.hpp"
#include "../TensorWrapper.hpp"
#include <cmath>

class BatchNormLayer : public Layer {
public:
    BatchNormLayer(int numFeatures, float momentum = 0.9, float epsilon = 1e-5)
        : numFeatures(numFeatures), momentum(momentum), epsilon(epsilon) {
        gamma = TensorOperations::ones(numFeatures);
        beta = TensorOperations::zeros(numFeatures);
        runningMean = TensorOperations::zeros(numFeatures);
        runningVariance = TensorOperations::ones(numFeatures);
    }

    Tensor forward(Tensor& input) override {
        Tensor& m = *mean;
        Tensor& v = *variance;
        Tensor& rMean = *runningMean;
        Tensor& rVar = *runningVariance;
        if (trainingMode) {
            m = input.mean(/*axis=*/0);
            variance = input.var(/*axis=*/0);

            normalizedInput = (input - m) / (v + epsilon).sqrt();

            runningMean = momentum * rMean + (1 - momentum) * m;
            runningVariance = momentum * rVar + (1 - momentum) * v;
        } else {
            normalizedInput = (input - rMean) / (rVar + epsilon).sqrt();
        }

        // Scale and shift
        output = (*gamma) * (*normalizedInput) + (*beta);
        return *output;
    }

    Tensor backward(Tensor& gradOutput) override {
        Tensor& nInput = *normalizedInput;
        
        gradGamma = gradOutput * nInput;
        gradGamma = (*gradGamma).sum(/*axis=*/0);
        
        gradBeta = gradOutput.sum(/*axis=*/0);

        // Calculate gradient with respect to input
        Tensor gradInput{gradOutput.shape()};
        if (trainingMode) {
            Tensor stdInv = 1.0 / ((*variance) + epsilon).sqrt();
            int batchSize = gradOutput.shape()[0];

            gradInput = (1.0 / batchSize) * (*gamma) * stdInv *
                        (batchSize * gradOutput - (*gradGamma) * nInput - (*gradBeta));
        } else {
            gradInput = (*gamma) * gradOutput / ((*runningVariance) + epsilon).sqrt();
        }

        return gradInput;
    }

    void updateWeights(Optimizer& optimizer) override {
    optimizer.optimize({std::ref(*gamma), std::ref(*beta)},
                       {std::ref(*gradGamma), std::ref(*gradBeta)});
}

private:
    int numFeatures;
    float momentum;
    float epsilon;

    TensorWrapper gamma;
    TensorWrapper beta;
    TensorWrapper runningMean;
    TensorWrapper runningVariance;

    TensorWrapper mean;
    TensorWrapper variance;
    TensorWrapper normalizedInput;
    TensorWrapper output;

    TensorWrapper gradGamma;
    TensorWrapper gradBeta;
};

#endif // BATCHNORM_LAYER_HPP

#ifndef BATCH_NORMALIZATION_LAYER_HPP
#define BATCH_NORMALIZATION_LAYER_HPP

#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/Layer.hpp"
#include "smart_dnn/tensor/Backend/Default/AdvancedTensorOperations.hpp"
#include <cmath>
#include <optional>
#include <iostream>

namespace sdnn {

class BatchNormalizationLayer : public Layer {
public:
    BatchNormalizationLayer(int numFeatures, float epsilon = 1e-5f, float momentum = 0.9f)
        : numFeatures(numFeatures), epsilon(epsilon), momentum(momentum), 
          gamma(Tensor({1, numFeatures}, 1.0f)), beta(Tensor({1, numFeatures}, 0.0f)),
          runningMean(Tensor({1, numFeatures}, 0.0f)), runningVariance(Tensor({1, numFeatures}, 1.0f)) {}

    Tensor forward(const Tensor& input) override {
        const auto& shape = input.shape();

        if (shape.rank() != 2 && shape.rank() != 4) {
            throw std::runtime_error("BatchNormalizationLayer: input tensor must have rank 2 or 4");
        }

        std::vector<size_t> reductionAxes = (shape.rank() == 4) ? std::vector<size_t>{0, 2, 3} : std::vector<size_t>{0};

        if (this->trainingMode) {
            batchMean = mean(input, reductionAxes);
            Tensor broadcastedMean = reshapeForBroadcast(*batchMean, shape);
            batchVariance = variance(input, broadcastedMean, reductionAxes);
            
            Tensor reshapedMean = reshapeForBroadcast(*batchMean, shape);
            Tensor reshapedVariance = reshapeForBroadcast(*batchVariance, shape);
            Tensor reshapedGamma = reshapeForBroadcast(gamma, shape);
            Tensor reshapedBeta = reshapeForBroadcast(beta, shape);

            normalizedInput = (input - reshapedMean) / sqrt(reshapedVariance + epsilon);
            Tensor output = (*normalizedInput * reshapedGamma) + reshapedBeta;

            runningMean = (momentum * runningMean) + ((1 - momentum) * reshape(*batchMean, runningMean.shape()));
            runningVariance = (momentum * runningVariance) + ((1 - momentum) * reshape(*batchVariance, runningVariance.shape()));
            return output;
        } else {
            Tensor reshapedMean = reshapeForBroadcast(runningMean, shape);
            Tensor reshapedVariance = reshapeForBroadcast(runningVariance, shape);
            Tensor reshapedGamma = reshapeForBroadcast(gamma, shape);
            Tensor reshapedBeta = reshapeForBroadcast(beta, shape);

            Tensor normalizedInput = (input - reshapedMean) / sqrt(reshapedVariance + epsilon);
            return (normalizedInput * reshapedGamma) + reshapedBeta;
        }
    }

    Tensor backward(const Tensor& gradOutput) override {
        if (!batchMean || !batchVariance || !normalizedInput) {
            throw std::runtime_error("BatchNormalizationLayer: forward must be called before backward");
        }

        const auto& inputShape = gradOutput.shape();
        
        int batchSize = inputShape[0];
        std::vector<size_t> reductionAxes = (inputShape.rank() == 4) ? std::vector<size_t>{0, 2, 3} : std::vector<size_t>{0};

        Tensor reshapedGamma = reshapeForBroadcast(gamma, inputShape);

        gammaGrad = sum(gradOutput * (*normalizedInput), reductionAxes);
        betaGrad = sum(gradOutput, reductionAxes);

        Tensor dNormalized = gradOutput * reshapedGamma;
        
        Tensor variance = reshapeForBroadcast(*batchVariance + epsilon, inputShape);
        Tensor stdDev = sqrt(variance);
        Tensor invStdDev = reciprocal(stdDev);
        
        Tensor dVariance = sum(dNormalized * (*normalizedInput), reductionAxes);
        dVariance = reshapeForBroadcast(dVariance, invStdDev.shape());
        dVariance = dVariance * -0.5f * invStdDev * invStdDev * invStdDev;
        dVariance = reshapeForBroadcast(dVariance, inputShape);
        
        Tensor dMean = sum(dNormalized * invStdDev * -1.0f, reductionAxes);
        dMean = reshapeForBroadcast(dMean, inputShape);

        Tensor dInput = dNormalized * invStdDev +
                        dVariance * 2.0f * (*normalizedInput) / static_cast<float>(batchSize) +
                        dMean / static_cast<float>(batchSize);

        return dInput;
    }

    void updateWeights(Optimizer& optimizer) override {
        if (!gammaGrad || !betaGrad) {
            throw std::runtime_error("BatchNormalizationLayer: gradients are not initialized");
        }

        // Ensure gamma and beta have the same shape as their gradients
        if (gamma.shape() != gammaGrad->shape()) {
            gamma = reshape(gamma, gammaGrad->shape());
        }
        if (beta.shape() != betaGrad->shape()) {
            beta = reshape(beta, betaGrad->shape());
        }

        optimizer.optimize({std::ref(gamma), std::ref(beta)},
                        {std::ref(*gammaGrad), std::ref(*betaGrad)});
    }

private:
    int numFeatures;
    float epsilon;
    float momentum;
    bool trainingMode = true;

    Tensor gamma;
    Tensor beta;
    Tensor runningMean;
    Tensor runningVariance;

    std::optional<Tensor> batchMean;
    std::optional<Tensor> batchVariance;
    std::optional<Tensor> normalizedInput;

    std::optional<Tensor> gammaGrad;
    std::optional<Tensor> betaGrad;

    Tensor reshapeForBroadcast(const Tensor& tensor, const Shape& targetShape) {
        std::vector<int> newShape;
        
        if (targetShape.rank() == 2) {
            newShape = {1, targetShape[1]};
        } else if (targetShape.rank() == 4) {
            newShape = {1, targetShape[1], 1, 1};
        } else {
            throw std::runtime_error("Unsupported target shape for broadcasting");
        }

        return reshape(tensor, Shape(newShape));
    }
};

} // namespace smart_dnn

#endif // BATCH_NORMALIZATION_LAYER_HPP
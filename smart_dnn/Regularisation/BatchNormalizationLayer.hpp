#ifndef BATCH_NORMALIZATION_LAYER_HPP
#define BATCH_NORMALIZATION_LAYER_HPP

#include "../Tensor/Tensor.hpp"
#include "../Layer.hpp"
#include <cmath>
#include <optional>
#include <iostream>

namespace smart_dnn {

template <typename T = float>
class BatchNormalizationLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    BatchNormalizationLayer(int numFeatures, T epsilon = 1e-5, T momentum = 0.9)
        : numFeatures(numFeatures), epsilon(epsilon), momentum(momentum), 
          gamma(TensorType({1, numFeatures}, T(1))), beta(TensorType({1, numFeatures}, T(0))),
          runningMean(TensorType({1, numFeatures}, T(0))), runningVariance(TensorType({1, numFeatures}, T(1))) {}

    TensorType forward(const TensorType& input) override {
        const auto& shape = input.getShape();

        if (shape.rank() != 2 && shape.rank() != 4) {
            throw std::runtime_error("BatchNormalizationLayer: input tensor must have rank 2 or 4");
        }

        std::vector<size_t> reductionAxes = (shape.rank() == 4) ? std::vector<size_t>{0, 2, 3} : std::vector<size_t>{0};

        if (this->trainingMode) {
            batchMean = AdvancedTensorOperations<T>::mean(input, reductionAxes);
            
            batchVariance = AdvancedTensorOperations<T>::variance(input, *batchMean, reductionAxes);

            TensorType reshapedMean = reshapeForBroadcast(*batchMean, shape);
            TensorType reshapedVariance = reshapeForBroadcast(*batchVariance, shape);
            TensorType reshapedGamma = reshapeForBroadcast(gamma, shape);
            TensorType reshapedBeta = reshapeForBroadcast(beta, shape);

            normalizedInput = (input - reshapedMean) / (reshapedVariance + epsilon).sqrt();
            TensorType output = (*normalizedInput * reshapedGamma) + reshapedBeta;

            runningMean = (momentum * runningMean) + ((1 - momentum) * (*batchMean));
            runningVariance = (momentum * runningVariance) + ((1 - momentum) * (*batchVariance));

            return output;
        } else {
            TensorType reshapedMean = reshapeForBroadcast(runningMean, shape);
            TensorType reshapedVariance = reshapeForBroadcast(runningVariance, shape);
            TensorType reshapedGamma = reshapeForBroadcast(gamma, shape);
            TensorType reshapedBeta = reshapeForBroadcast(beta, shape);

            TensorType normalizedInput = (input - reshapedMean) / (reshapedVariance + epsilon).sqrt();
            return (normalizedInput * reshapedGamma) + reshapedBeta;
        }
        }

    TensorType backward(const TensorType& gradOutput) override {
        if (!batchMean || !batchVariance || !normalizedInput) {
            throw std::runtime_error("BatchNormalizationLayer: forward must be called before backward");
        }

        const auto& inputShape = gradOutput.getShape();
        
        int batchSize = inputShape[0];
        std::vector<size_t> reductionAxes = (inputShape.rank() == 4) ? std::vector<size_t>{0, 2, 3} : std::vector<size_t>{0};

        TensorType reshapedGamma = reshapeForBroadcast(gamma, inputShape);

        TensorType dGamma = AdvancedTensorOperations<T>::sum(gradOutput * (*normalizedInput), reductionAxes);
        TensorType dBeta = AdvancedTensorOperations<T>::sum(gradOutput, reductionAxes);

        TensorType dNormalized = gradOutput * reshapedGamma;
        
        TensorType variance = reshapeForBroadcast(*batchVariance + epsilon, inputShape);
        TensorType stdDev = variance.sqrt();
        TensorType invStdDev = AdvancedTensorOperations<T>::reciprocal(stdDev);
        
        TensorType dVariance = AdvancedTensorOperations<T>::sum(dNormalized * (*normalizedInput), reductionAxes);
        dVariance = reshapeForBroadcast(dVariance, invStdDev.getShape());
        dVariance = dVariance * T(-0.5) * invStdDev * invStdDev * invStdDev;
        dVariance = reshapeForBroadcast(dVariance, inputShape);
        
        TensorType dMean = AdvancedTensorOperations<T>::sum(dNormalized * invStdDev * T(-1), reductionAxes);
        dMean = reshapeForBroadcast(dMean, inputShape);

        
        TensorType dInput = dNormalized * invStdDev +
                            dVariance * T(2) * (*normalizedInput) / T(batchSize) +
                            dMean / T(batchSize);

        gammaGrad = dGamma;
        betaGrad = dBeta;

        return dInput;
    }

    void updateWeights(Optimizer<T>& optimizer) override {
        if (!gammaGrad || !betaGrad) {
            throw std::runtime_error("BatchNormalizationLayer: gradients are not initialized");
        }

        optimizer.optimize({std::ref(gamma), std::ref(beta)},
                           {std::ref(*gammaGrad), std::ref(*betaGrad)});
    }

private:
    int numFeatures;
    T epsilon;
    T momentum;

    TensorType gamma;
    TensorType beta;
    TensorType runningMean;
    TensorType runningVariance;

    std::optional<TensorType> batchMean;
    std::optional<TensorType> batchVariance;
    std::optional<TensorType> normalizedInput;

    std::optional<TensorType> gammaGrad;
    std::optional<TensorType> betaGrad;

    TensorType reshapeForBroadcast(const TensorType& tensor, const Shape& targetShape) {
        std::vector<int> newShape(targetShape.rank(), 1);
        newShape[1] = tensor.getShape()[1];  // Preserve the channel dimension
        
        // For 2D input, we need to keep the first dimension as 1 to allow broadcasting
        if (targetShape.rank() == 2) {
            newShape[0] = 1;
        }

        TensorType reshaped = AdvancedTensorOperations<T>::reshape(tensor, Shape(newShape));
        return reshaped;
    }
};

} // namespace smart_dnn

#endif // BATCH_NORMALIZATION_LAYER_HPP
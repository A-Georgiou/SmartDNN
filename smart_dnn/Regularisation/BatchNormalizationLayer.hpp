#ifndef BATCH_NORMALIZATION_LAYER_HPP
#define BATCH_NORMALIZATION_LAYER_HPP

#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Layer.hpp"
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

            // Batch statistics are (1, C, 1, 1) for 4D input; match the (1, C) running statistics.
            runningMean = (momentum * runningMean) + ((1 - momentum) * AdvancedTensorOperations<T>::reshape(*batchMean, runningMean.getShape()));
            runningVariance = (momentum * runningVariance) + ((1 - momentum) * AdvancedTensorOperations<T>::reshape(*batchVariance, runningVariance.getShape()));

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
        std::vector<size_t> reductionAxes = (inputShape.rank() == 4) ? std::vector<size_t>{0, 2, 3} : std::vector<size_t>{0};

        TensorType reshapedGamma = reshapeForBroadcast(gamma, inputShape);

        TensorType dGamma = AdvancedTensorOperations<T>::sum(gradOutput * (*normalizedInput), reductionAxes);
        TensorType dBeta = AdvancedTensorOperations<T>::sum(gradOutput, reductionAxes);

        // dx = (g - mean(g) - xhat * mean(g * xhat)) / sqrt(var + eps), averaging over each channel's N (2D) or N * H * W (4D) values.
        TensorType dNormalized = gradOutput * reshapedGamma;
        const T count = T(inputShape.size() / numFeatures);
        TensorType invStdDev = AdvancedTensorOperations<T>::reciprocal(reshapeForBroadcast(*batchVariance + epsilon, inputShape).sqrt());
        TensorType meanDNormalized = reshapeForBroadcast(AdvancedTensorOperations<T>::sum(dNormalized, reductionAxes), inputShape) * (T(1) / count);
        TensorType meanDNormalizedXhat = reshapeForBroadcast(AdvancedTensorOperations<T>::sum(dNormalized * (*normalizedInput), reductionAxes), inputShape) * (T(1) / count);
        TensorType dInput = invStdDev * (dNormalized - meanDNormalized - (*normalizedInput) * meanDNormalizedXhat);

        gammaGrad = AdvancedTensorOperations<T>::reshape(dGamma, gamma.getShape());
        betaGrad = AdvancedTensorOperations<T>::reshape(dBeta, beta.getShape());

        return dInput;
    }

    std::vector<Parameter<T>> parameters() override {
        if (!gammaGrad || !betaGrad) {
            throw std::runtime_error("BatchNormalizationLayer: gradients are not initialized");
        }

        return {{gamma, *gammaGrad}, {beta, *betaGrad}};
    }

    void save(std::ostream& os) const override {
        saveTensor(os, gamma);
        saveTensor(os, beta);
        saveTensor(os, runningMean);
        saveTensor(os, runningVariance);
    }

    void load(std::istream& is) override {
        loadTensor(is, gamma);
        loadTensor(is, beta);
        loadTensor(is, runningMean);
        loadTensor(is, runningVariance);
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
        newShape[1] = numFeatures;  // Reductions drop the summed axes, so take the channel count from the layer
        
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
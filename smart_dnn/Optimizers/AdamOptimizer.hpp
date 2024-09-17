#ifndef ADAM_OPTIMIZER_HPP
#define ADAM_OPTIMIZER_HPP

#include <unordered_map>
#include <cmath>
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/Optimizer.hpp"

namespace sdnn {

struct AdamOptions {
    float learningRate = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-8;
    float l1Strength = 0;
    float l2Strength = 0;
    float decay = 0;
    int batchSize = 1;
};

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(const AdamOptions& options = {})
        : initialLearningRate(options.learningRate),
          learningRate(options.learningRate),
          beta1(options.beta1),
          beta2(options.beta2),
          epsilon(options.epsilon),
          l1Strength(options.l1Strength),
          l2Strength(options.l2Strength),
          decay(options.decay),
          iterations(0),
          batchSize(options.batchSize) {}

    void optimize(const std::vector<std::reference_wrapper<Tensor>>& weights,
                  const std::vector<std::reference_wrapper<Tensor>>& gradients,
                  float learningRateOverride = -1) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Weights and gradients size mismatch!");
        }

        updateLearningRate(learningRateOverride);

        for (size_t i = 0; i < weights.size(); ++i) {
            updateTensor(weights[i].get(), gradients[i].get());
        }
    }

private:
    float initialLearningRate;
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float l1Strength;
    float l2Strength;
    float decay;
    int iterations;
    int batchSize;
    std::unordered_map<size_t, Tensor> m; // First moment estimate 
    std::unordered_map<size_t, Tensor> v; // Second moment estimate

    void updateLearningRate(float learningRateOverride) {
        iterations++;
        if (learningRateOverride <= 0) {
            learningRate = initialLearningRate * (decay > 0 ? (1 / (1 + decay * iterations)) : 1);
        } else {
            learningRate = learningRateOverride;
        }
    }

    void updateTensor(Tensor& weight, const Tensor& gradient) {
        size_t key = reinterpret_cast<size_t>(&weight);
        initializeMomentEstimates(key, weight.shape());

        float beta1Power = std::max(static_cast<float>(std::pow(beta1, iterations)), std::numeric_limits<float>::min());
        float beta2Power = std::max(static_cast<float>(std::pow(beta2, iterations)), std::numeric_limits<float>::min());
        float out = (learningRate * std::sqrt(1 - beta2Power) / (1 - beta1Power));
        Tensor alphaT = Tensor(Shape{1}, out, weight.type());

        size_t size = weight.shape().size();
        Tensor mData = m.at(key);
        Tensor vData = v.at(key);

        #pragma omp parallel for if(size > 1000)
        for (size_t j = 0; j < size; ++j) {
            Tensor weightElement = weight[j];
            Tensor gradientElement = gradient[j];
            Tensor mElement = mData[j];
            Tensor vElement = vData[j];

            updateParameter(weightElement, gradientElement, mElement, vElement, alphaT);
        }
    }

    void initializeMomentEstimates(size_t key, const Shape& shape) {
        if (m.find(key) == m.end()) {
            m.emplace(key, zeros(shape, dtype::f32));
            v.emplace(key, zeros(shape, dtype::f32));
        }
    }

    void updateParameter(Tensor& weight, const Tensor& gradient, Tensor& mValue, Tensor& vValue, const Tensor& alphaT) {
        Tensor averagedGradient = gradient / batchSize;

        // Update biased first moment estimate
        mValue = beta1 * mValue + (1 - beta1) * averagedGradient;

        // Update biased second moment estimate
        vValue = beta2 * vValue + (1 - beta2) * averagedGradient * averagedGradient;

        // Compute update
        Tensor mHat = mValue / (1 - std::pow(beta1, iterations));
        Tensor vHat = vValue / (1 - std::pow(beta2, iterations));
        Tensor update = alphaT * mHat / (sqrt(vHat) + epsilon);

        // Apply L1 and L2 regularization
        if (l1Strength > 0) {
            Tensor l1Grad = apply(weight, [this](double& x) { x = (x > 0 ? 1 : (x < 0 ? -1 : 0)); });
            update += learningRate * l1Strength * l1Grad;
        }
        if (l2Strength > 0) {
            update += learningRate * l2Strength * weight;
        }

        // Apply the update
        weight -= update;
    }
};

} // namespace sdnn

#endif // ADAM_OPTIMIZER_HPP
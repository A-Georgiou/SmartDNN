#ifndef ADAM_OPTIMIZER_HPP
#define ADAM_OPTIMIZER_HPP

#include <unordered_map>
#include <cmath>
#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Optimizer.hpp"

namespace smart_dnn {

template <typename T = float>
struct AdamOptions {
    T learningRate = T(0.001);
    T beta1 = T(0.9);
    T beta2 = T(0.999);
    T epsilon = T(1e-8);
    T l1Strength = T(0);
    T l2Strength = T(0);
    T decay = T(0);
    int batchSize = 1;
};

template <typename T=float>
class AdamOptimizer : public Optimizer<T> {
public:
    AdamOptimizer(const AdamOptions<T>& options = {})
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

    void optimize(const std::vector<std::reference_wrapper<Tensor<T>>>& weights,
                  const std::vector<std::reference_wrapper<Tensor<T>>>& gradients,
                  T learningRateOverride = T(-1)) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Weights and gradients size mismatch!");
        }

        updateLearningRate(learningRateOverride);

        for (size_t i = 0; i < weights.size(); ++i) {
            updateTensor(weights[i].get(), gradients[i].get());
        }
    }

private:
    T initialLearningRate;
    T learningRate;
    T beta1;
    T beta2;
    T epsilon;
    T l1Strength;
    T l2Strength;
    T decay;
    int iterations;
    int batchSize;
    std::unordered_map<size_t, Tensor<T>> m; // First moment estimate 
    std::unordered_map<size_t, Tensor<T>> v; // Second moment estimate

    void updateLearningRate(T learningRateOverride) {
        iterations++;
        if (learningRateOverride <= T(0)) {
            learningRate = initialLearningRate * (decay > 0 ? (T(1) / (T(1) + decay * iterations)) : 1);
        } else {
            learningRate = learningRateOverride;
        }
    }

    void updateTensor(Tensor<T>& weight, const Tensor<T>& gradient) {
        size_t key = reinterpret_cast<size_t>(&weight);
        initializeMomentEstimates(key, weight.getShape());

        T beta1Power = std::max(T(std::pow(beta1, iterations)), std::numeric_limits<T>::min());
        T beta2Power = std::max(T(std::pow(beta2, iterations)), std::numeric_limits<T>::min());
        T alphaT = learningRate * std::sqrt(T(1) - beta2Power) / (T(1) - beta1Power);

        size_t size = weight.getShape().size();
        T* weightData = weight.getData().data();
        const T* gradientData = gradient.getData().data();
        T* mData = m.at(key).getData().data();
        T* vData = v.at(key).getData().data();

        #pragma omp parallel for if(size > 1000)
        for (size_t j = 0; j < size; ++j) {
            updateParameter(weightData[j], gradientData[j], mData[j], vData[j], alphaT);
        }
    }

    void initializeMomentEstimates(size_t key, const Shape& shape) {
        if (m.find(key) == m.end()) {
            m.emplace(key, Tensor<T>(shape, T(0)));
            v.emplace(key, Tensor<T>(shape, T(0)));
        }
    }

    void updateParameter(T& weight, const T& gradient, T& mValue, T& vValue, T alphaT) {
        // Average the gradient over the batch
        T averagedGradient = gradient / static_cast<T>(batchSize);

        // Update biased first moment estimate
        mValue = beta1 * mValue + (T(1) - beta1) * averagedGradient;

        // Update biased second moment estimate
        vValue = beta2 * vValue + (T(1) - beta2) * averagedGradient * averagedGradient;

        // Compute update
        T mHat = mValue / (T(1) - std::pow(beta1, iterations));
        T vHat = vValue / (T(1) - std::pow(beta2, iterations));
        T update = alphaT * mHat / (std::sqrt(vHat) + epsilon);

        // Apply L1 and L2 regularization
        if (l1Strength > T(0)) {
            T l1Grad = (weight > T(0)) ? T(1) : ((weight < T(0)) ? T(-1) : T(0));
            update += learningRate * l1Strength * l1Grad;
        }
        if (l2Strength > T(0)) {
            update += learningRate * l2Strength * weight;
        }

        // Apply the update
        weight -= update;
    }
};

} // namespace smart_dnn

#endif // ADAM_OPTIMIZER_HPP
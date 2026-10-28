#ifndef ADAGRAD_OPTIMIZER_HPP
#define ADAGRAD_OPTIMIZER_HPP

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Optimizer.hpp"

namespace smart_dnn {

template <typename T = float>
struct AdaGradOptions {
    T learningRate = T(0.01);
    T epsilon = T(1e-8);
};

template <typename T = float>
class AdaGradOptimizer : public Optimizer<T> {
public:
    AdaGradOptimizer(const AdaGradOptions<T>& options = {})
        : learningRate(options.learningRate), epsilon(options.epsilon) {}

    void optimize(const std::vector<std::reference_wrapper<Tensor<T>>>& weights,
                  const std::vector<std::reference_wrapper<Tensor<T>>>& gradients,
                  T learningRateOverride = T(-1)) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Weights and gradients size mismatch!");
        }

        const T currentLR = learningRateOverride > T(0) ? learningRateOverride : learningRate;
        for (size_t i = 0; i < weights.size(); ++i) {
            updateTensor(weights[i].get(), gradients[i].get(), currentLR);
        }
    }

private:
    void updateTensor(Tensor<T>& weight, const Tensor<T>& gradient, T currentLR) {
        auto state = accumulatedSquaredGradients.find(&weight);
        if (state == accumulatedSquaredGradients.end()) {
            state = accumulatedSquaredGradients.emplace(
                &weight, Tensor<T>(weight.getShape(), T(0))).first;
        }

        T* weightData = weight.getData().data();
        const T* gradientData = gradient.getData().data();
        T* accumulatedData = state->second.getData().data();
        const size_t size = weight.getShape().size();

        for (size_t i = 0; i < size; ++i) {
            accumulatedData[i] += gradientData[i] * gradientData[i];
            weightData[i] -= currentLR * gradientData[i] /
                             (std::sqrt(accumulatedData[i]) + epsilon);
        }
    }

    T learningRate;
    T epsilon;
    std::unordered_map<const Tensor<T>*, Tensor<T>> accumulatedSquaredGradients;
};

} // namespace smart_dnn

#endif // ADAGRAD_OPTIMIZER_HPP

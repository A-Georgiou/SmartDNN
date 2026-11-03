#ifndef ADADELTA_OPTIMIZER_HPP
#define ADADELTA_OPTIMIZER_HPP

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Optimizer.hpp"

namespace smart_dnn {

template <typename T = float>
struct AdaDeltaOptions {
    T learningRate = T(1);
    T decay = T(0.95);
    T epsilon = T(1e-6);
};

template <typename T = float>
class AdaDeltaOptimizer : public Optimizer<T> {
public:
    AdaDeltaOptimizer(const AdaDeltaOptions<T>& options = {})
        : learningRate(options.learningRate),
          decay(options.decay),
          epsilon(options.epsilon) {}

    void optimize(const std::vector<std::reference_wrapper<Tensor<T>>>& weights,
                  const std::vector<std::reference_wrapper<Tensor<T>>>& gradients,
                  T learningRateOverride = T(-1)) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Weights and gradients size mismatch!");
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            if (weights[i].get().getShape() != gradients[i].get().getShape()) {
                throw std::invalid_argument("Weight and gradient shape mismatch!");
            }
        }

        T currentLR = (learningRateOverride > T(0)) ? learningRateOverride : learningRate;
        for (size_t i = 0; i < weights.size(); ++i) {
            updateTensor(weights[i].get(), gradients[i].get(), currentLR);
        }
    }

private:
    struct TensorState {
        explicit TensorState(const Shape& shape)
            : squaredGradients(shape, T(0)),
              squaredUpdates(shape, T(0)) {}

        Tensor<T> squaredGradients;
        Tensor<T> squaredUpdates;
    };

    void updateTensor(Tensor<T>& weight, const Tensor<T>& gradient, T currentLR) {
        auto state = states.try_emplace(&weight, weight.getShape()).first;
        if (state->second.squaredGradients.getShape() != weight.getShape()) {
            state->second = TensorState(weight.getShape());
        }
        auto& weightData = weight.getData();
        const auto& gradientData = gradient.getData();
        auto& squaredGradients = state->second.squaredGradients.getData();
        auto& squaredUpdates = state->second.squaredUpdates.getData();

        for (size_t i = 0; i < weightData.size(); ++i) {
            T grad = gradientData[i];
            squaredGradients[i] = decay * squaredGradients[i] + (T(1) - decay) * grad * grad;
            T update = std::sqrt(squaredUpdates[i] + epsilon) /
                       std::sqrt(squaredGradients[i] + epsilon) * grad;
            squaredUpdates[i] = decay * squaredUpdates[i] + (T(1) - decay) * update * update;
            weightData[i] -= currentLR * update;
        }
    }

    T learningRate;
    T decay;
    T epsilon;
    std::unordered_map<const Tensor<T>*, TensorState> states;
};

} // namespace smart_dnn

#endif // ADADELTA_OPTIMIZER_HPP

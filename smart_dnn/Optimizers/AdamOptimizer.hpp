#ifndef ADAM_OPTIMIZER_HPP
#define ADAM_OPTIMIZER_HPP

#include <unordered_map>
#include "../Tensor/Tensor.hpp"
#include "../Optimizer.hpp"
#include "../TensorOperations.hpp"

namespace smart_dnn {

template <typename T = float>
struct AdamOptions {
    T learningRate = T(0.001);
    T beta1 = T(0.9);
    T beta2 = T(0.999);
    T epsilon = T(1e-8);
    T l1_strength = T(0);
    T l2_strength = T(0);
};

template <typename T=float>
class AdamOptimizer : public Optimizer<T> {
public:
    AdamOptimizer(AdamOptions<T> options = {})
        : learningRate(options.learningRate), beta1(options.beta1), beta2(options.beta2),
          epsilon(options.epsilon), l1_strength(options.l1_strength), l2_strength(options.l2_strength), t(0) {}

    void optimize(const std::vector<std::reference_wrapper<Tensor<T>>>& weights, const std::vector<std::reference_wrapper<Tensor<T>>>& gradients, T learningRateOverride = T(-1)) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Error: weights and gradients size mismatch!");
        }

        T lr = (learningRateOverride > T(0)) ? learningRateOverride : learningRate;
        t += 1;

        for (size_t i = 0; i < weights.size(); ++i) {
            Tensor<T>& w = weights[i].get();
            const Tensor<T>& g = gradients[i].get();

            size_t key = reinterpret_cast<size_t>(&w);

            if (m.find(key) == m.end()) {
                m.emplace(key, Tensor<T>(w.getShape(), T(0)));
                v.emplace(key, Tensor<T>(w.getShape(), T(0)));
            }

            Tensor<T>& m_v = m.at(key);
            Tensor<T>& v_v = v.at(key);

            // Update biased first moment estimate
            m.emplace(key, beta1 * m_v + (T(1) - beta1) * g);

            // Update biased second moment estimate
            v.emplace(key, beta2 * v_v + (T(1) - beta2) * g * g);

            // Compute bias-corrected first and second moment estimates
            Tensor<T> m_hat = m_v / (T(1) - std::pow(beta1, t));
            Tensor<T> v_hat = v_v / (T(1) - std::pow(beta2, t));

            // Apply L1 and L2 regularization
            if (l1_strength > T(0)) {
                w.apply([](T x) { return x > T(0) ? T(1) : T(-1); });
                w -= lr * l1_strength * w;
            }
            if (l2_strength > T(0)) {
                w -= lr * l2_strength * w;
            }

            // Update weights with Adam
            w -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }
    }

    void save(std::ostream& os) const override {
        // Implement saving state logic if needed
    }

    void load(std::istream& is) override {
        // Implement loading state logic if needed
    }

private:
    T learningRate;
    T beta1;
    T beta2;
    T epsilon;
    int t;
    T l1_strength;
    T l2_strength;
    
    std::unordered_map<size_t, Tensor<T>> m; // First moment estimate 
    std::unordered_map<size_t, Tensor<T>> v; // Second moment estimate
};

} // namespace smart_dnn

#endif // ADAM_OPTIMIZER_HPP
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

    void optimize(const std::vector<std::reference_wrapper<Tensor<T>>>& weights,
                const std::vector<std::reference_wrapper<Tensor<T>>>& gradients,
                T learningRateOverride = T(-1)) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Error: weights and gradients size mismatch!");
        }

        T lr = (learningRateOverride > T(0)) ? learningRateOverride : learningRate;
        t += 1;
        T beta1_t = std::pow(beta1, t);
        T beta2_t = std::pow(beta2, t);
        T alpha = lr * std::sqrt(T(1) - beta2_t) / (T(1) - beta1_t);

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

            size_t size = w.getShape().size();
            T* w_data = w.getData().data();
            const T* g_data = g.getData().data();
            T* m_data = m_v.getData().data();
            T* v_data = v_v.getData().data();

            #pragma omp parallel for if(size > 1000)
            for (size_t j = 0; j < size; ++j) {
                // Update biased first moment estimate
                m_data[j] = beta1 * m_data[j] + (T(1) - beta1) * g_data[j];

                // Update biased second moment estimate
                v_data[j] = beta2 * v_data[j] + (T(1) - beta2) * g_data[j] * g_data[j];

                // Compute update
                T m_hat = m_data[j] / (T(1) - beta1_t);
                T v_hat = v_data[j] / (T(1) - beta2_t);
                T update = alpha * m_hat / (std::sqrt(v_hat) + epsilon);

                // Apply L1 and L2 regularization
                if (l1_strength > T(0)) {
                    if (w_data[j] > T(0)) {
                        update += lr * l1_strength;
                    } else if (w_data[j] < T(0)) {
                        update -= lr * l1_strength;
                    }
                }
                if (l2_strength > T(0)) {
                    update += lr * l2_strength * w_data[j];
                }

                // Apply the update
                w_data[j] -= update;
            }
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
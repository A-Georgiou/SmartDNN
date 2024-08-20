#include <unordered_map>
#include "../Tensor.hpp"
#include "Optimizer.hpp"
#include "../TensorOperations.hpp"

struct AdamOptions {
    float learningRate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float l1_strength = 0.0f;
    float l2_strength = 0.0f;
};

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(AdamOptions options = {})
        : learningRate(options.learningRate), beta1(options.beta1), beta2(options.beta2),
          epsilon(options.epsilon), l1_strength(options.l1_strength), l2_strength(options.l2_strength), t(0) {}

    void optimize(const std::vector<std::reference_wrapper<Tensor>>& weights, const std::vector<std::reference_wrapper<Tensor>>& gradients, float learningRateOverride = -1.0f) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Error: weights and gradients size mismatch!");
        }

        float lr = (learningRateOverride > 0) ? learningRateOverride : learningRate;
        t += 1;

        for (size_t i = 0; i < weights.size(); ++i) {
            Tensor& w = weights[i].get();
            const Tensor& g = gradients[i].get();

            size_t key = reinterpret_cast<size_t>(&w);

            if (m.find(key) == m.end()) {
                m[key] = Tensor(w.shape(), 0.0f);
                v[key] = Tensor(w.shape(), 0.0f);
            }

            // Update biased first moment estimate
            m[key] = beta1 * m[key] + (1.0f - beta1) * g;

            // Update biased second moment estimate
            v[key] = beta2 * v[key] + (1.0f - beta2) * g * g;

            // Compute bias-corrected first and second moment estimates
            Tensor m_hat = m[key] / (1.0f - std::pow(beta1, t));
            Tensor v_hat = v[key] / (1.0f - std::pow(beta2, t));

            // Apply L1 and L2 regularization
            if (l1_strength > 0.0f) {
                w -= lr * l1_strength * w.apply([](float x) { return x > 0 ? 1.0f : -1.0f; });
            }
            if (l2_strength > 0.0f) {
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
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float l1_strength;
    float l2_strength;
    
    std::unordered_map<size_t, Tensor> m; // First moment estimate
    std::unordered_map<size_t, Tensor> v; // Second moment estimate
};

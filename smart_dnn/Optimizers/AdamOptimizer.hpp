#include <unordered_map>
#include "../Tensor.hpp"
#include "Optimizer.hpp"
#include "../TensorOperations.hpp"

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void optimize(const std::vector<std::reference_wrapper<Tensor>>& weights, const std::vector<std::reference_wrapper<Tensor>>& gradients, float learningRateOverride = -1.0f) override {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Error: weights and gradients size mismatch!");
        }

        float lr = (learningRateOverride > 0) ? learningRateOverride : learningRate;
        t += 1;

        for (size_t i = 0; i < weights.size(); ++i) {
            Tensor& w = weights[i].get();
            const Tensor& g = gradients[i].get();

            // Use the address of the weight tensor as a unique key
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

            // Update weights
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
    
    std::unordered_map<size_t, Tensor> m; // First moment estimate
    std::unordered_map<size_t, Tensor> v; // Second moment estimate
};

#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "../Tensor.hpp"
#include "Optimizer.hpp"

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void optimize(std::vector<Tensor>& weights, std::vector<Tensor>& gradients, float learningRateOverride = -1.0f) override {
        if (m.empty()) {
            initialize(weights);
        }

        float lr = (learningRateOverride > 0) ? learningRateOverride : learningRate;
        t += 1;

        for (size_t i = 0; i < weights.size(); ++i) {
            Tensor& w = weights[i];
            Tensor& g = gradients[i];

            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;

            // Update biased second moment estimate
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

            // Compute bias-corrected first and second moment estimates
            Tensor m_hat = m[i] / (1.0f - std::pow(beta1, t));
            Tensor v_hat = v[i] / (1.0f - std::pow(beta2, t));

            // Update weights
            w -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }
    }

    void save(std::ostream& os) const override {
        std::cout << "Saving Adam optimizer -- boilerplate" << std::endl;
    }

    void load(std::istream& is) override {
        std::cout << "Loading Adam optimizer -- boilerplate" << std::endl;
    }

private:
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    std::vector<Tensor> m;
    std::vector<Tensor> v;

    void initialize(const std::vector<Tensor>& weights) {
        m.resize(weights.size());
        v.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            m[i] = Tensor(weights[i].shape(), 0.0f);
            v[i] = Tensor(weights[i].shape(), 0.0f);
        }
    }

    void saveTensorVector(std::ostream& os, const std::vector<Tensor>& tensorVector) const {
        return;
    }

    void loadTensorVector(std::istream& is, std::vector<Tensor>& tensorVector) {
        return;
    }
};
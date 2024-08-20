#ifndef CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include "Loss.hpp"
#include "Tensor.hpp"
#include <cmath>

class CategoricalCrossEntropyLoss : public Loss {
public:
    ~CategoricalCrossEntropyLoss() override = default;

    float compute(const Tensor& prediction, const Tensor& target) override {
        if (prediction.shape().size() != target.shape().size()) {
            throw std::invalid_argument("Prediction and target tensors must have the same shape, prediction shape: " + prediction.shape().toString() + ", target shape: " + target.shape().toString());
        }

        float loss = 0.0f;
        const std::vector<float>& predData = prediction.getData();
        const std::vector<float>& targetData = target.getData();

        for (size_t i = 0; i < predData.size(); ++i) {
            if (targetData[i] == 1.0f) { 
                loss -= std::log(predData[i] + 1e-15f); // Small episilon added to avoid log(0)
            }
        }

        return loss / prediction.shape()[0]; 
    }

    Tensor gradient(const Tensor& prediction, const Tensor& target) override {
        if (prediction.shape().size() != target.shape().size()) {
            throw std::invalid_argument("Prediction and target tensors must have the same shape, prediction shape: " + prediction.shape().toString() + ", target shape: " + target.shape().toString());
        }

        Tensor grad(prediction.shape());
        std::vector<float>& gradData = grad.getData();
        const std::vector<float>& predData = prediction.getData();
        const std::vector<float>& targetData = target.getData();

        for (size_t i = 0; i < predData.size(); ++i) {
            gradData[i] = predData[i] - targetData[i]; 
        }

        return grad;
    }

    void save(std::ostream& os) const override {
        return;
    }

    void load(std::istream& is) override {
        return;
    }
};

#endif // CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

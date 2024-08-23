#ifndef CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include "../Loss.hpp"
#include "../Tensor.hpp"
#include "../TensorOperations.hpp"
#include <cmath>
#include <numeric>

class CategoricalCrossEntropyLoss : public Loss {
public:
    float compute(const Tensor& prediction, const Tensor& target) override {
        if (prediction.shape() != target.shape()) {
            throw std::invalid_argument("Prediction and target shapes must match, mismatch: " + std::to_string(prediction.shape().size()) + " != " + std::to_string(target.shape().size()));
        }

        const float* predData = prediction.getData();
        const float* targetData = target.getData();
        int size = prediction.shape().size();

        float loss = 0.0f;
        for (int i = 0; i < size; ++i) {
            loss -= targetData[i] * std::log(predData[i] + 1e-7f);
        }

        return loss / prediction.shape()[0];  // Normalize by batch size
    }

    Tensor gradient(const Tensor& prediction, const Tensor& target) override {
        if (prediction.shape() != target.shape()) {
            throw std::invalid_argument("Prediction and target shapes must match");
        }

        Tensor grad(prediction.shape());
        float* gradData = grad.getData();
        const float* predData = prediction.getData();
        const float* targetData = target.getData();
        int size = prediction.shape().size();

        for (int i = 0; i < size; ++i) {
            gradData[i] = (predData[i] - targetData[i]) / prediction.shape()[0];
        }

        return grad;
    }

    void save(std::ostream& os) const override {
        // No parameters to save for this loss function
    }

    void load(std::istream& is) override {
        // No parameters to load for this loss function
    }
};

#endif // CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

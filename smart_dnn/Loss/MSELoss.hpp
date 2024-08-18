#ifndef MSELOSS_HPP
#define MSELOSS_HPP

#include "Loss.hpp"
#include "Tensor.hpp"
#include <cmath>

class MSELoss : public Loss {
public:
    ~MSELoss() override = default;

    float compute(const Tensor& prediction, const Tensor& target) override {
        if (prediction.shape() != target.shape()) {
            throw std::invalid_argument("Shapes of prediction and target do not match.");
        }

        Tensor diff = prediction - target;
        float mse = ((diff * diff).sum()) / diff.shape().size();
        return mse;
    }

    Tensor gradient(const Tensor& prediction, const Tensor& target) override {
        if (prediction.shape() != target.shape()) {
            throw std::invalid_argument("Shapes of prediction and target do not match.");
        }

        Tensor grad = 2 * (prediction - target) / prediction.shape().size();
        return grad;
    }

    void save(std::ostream& os) const override {
        return;
    }

    void load(std::istream& is) override {
        return;
    }
};

#endif // MSELOSS_HPP
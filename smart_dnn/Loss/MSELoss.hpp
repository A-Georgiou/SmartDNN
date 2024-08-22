#ifndef MSELOSS_HPP
#define MSELOSS_HPP

#include "../Loss.hpp"
#include "../Tensor.hpp"
#include <cmath>

class MSELoss : public Loss {
public:
    ~MSELoss() override = default;

    float compute(const Tensor& prediction, const Tensor& target) override {
        Tensor reshapedTarget = target;

        if (prediction.shape() != target.shape()) {
            if (target.shape().rank() == 1 && target.shape()[0] == prediction.shape()[0]) {
                reshapedTarget = target.reshape(prediction.shape());
            } else {
                throw std::invalid_argument("Shapes of prediction and target do not match and cannot be broadcast.");
            }
        }

        Tensor diff = prediction - reshapedTarget;
        float mse = ((diff * diff).sum()) / diff.shape().size();
        return mse;
    }

    Tensor gradient(const Tensor& prediction, const Tensor& target) override {
        Tensor reshapedTarget = target;

        if (prediction.shape() != target.shape()) {
            if (target.shape().rank() == 1 && target.shape()[0] == prediction.shape()[0]) {
                reshapedTarget = target.reshape(prediction.shape());
            } else {
                throw std::invalid_argument("Shapes of prediction and target do not match and cannot be broadcast.");
            }
        }

        Tensor grad = 2 * (prediction - reshapedTarget) / prediction.shape().size();
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

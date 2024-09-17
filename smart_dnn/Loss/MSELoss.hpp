#ifndef MSELOSS_HPP
#define MSELOSS_HPP

#include "smart_dnn/Loss.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include <cmath>

namespace sdnn {

class MSELoss : public Loss {
public:
    ~MSELoss() override = default;

    Tensor compute(const Tensor& prediction, const Tensor& target) override {
        Tensor reshapedTarget = target;

        if (prediction.shape() != target.shape()) {
            if (target.shape().rank() == 1 && target.shape()[0] == prediction.shape()[0]) {
                reshapedTarget = reshape(target, prediction.shape());
            } else {
                throw std::invalid_argument("Shapes of prediction and target do not match and cannot be broadcast.");
            }
        }

        Tensor diff = prediction - reshapedTarget;
        Tensor mse = sum((diff * diff)) / diff.shape().size();
        return mse;
    }

    Tensor gradient(const Tensor& prediction, const Tensor& target) override {
        Tensor reshapedTarget = target;

        if (prediction.shape() != target.shape()) {
            if (target.shape().rank() == 1 && target.shape()[0] == prediction.shape()[0]) {
                reshapedTarget = reshape(target, prediction.shape());
            } else {
                throw std::invalid_argument("Shapes of prediction and target do not match and cannot be broadcast.");
            }
        }

        Tensor grad = 2 * (prediction - reshapedTarget) / prediction.shape().size();
        return grad;
    }
};

} // namespace sdnn

#endif // MSELOSS_HPP

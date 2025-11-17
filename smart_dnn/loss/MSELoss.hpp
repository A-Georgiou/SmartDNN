#ifndef MSELOSS_HPP
#define MSELOSS_HPP

#include "smart_dnn/Loss.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include <cmath>

namespace sdnn {

class MSELoss : public Loss {
public:
    ~MSELoss() override = default;

    Tensor compute(const Tensor& prediction, const Tensor& target) override {
        Tensor reshapedTarget = target;

        if (!ShapeOperations::areBroadcastable(prediction.shape(), target.shape())) {
            throw std::invalid_argument("Shapes of prediction and target are not broadcast-compatible.");
        }

        Tensor diff = prediction - reshapedTarget;
        return mean((diff * diff));
    }

    Tensor gradient(const Tensor& prediction, const Tensor& target) override {
        Tensor reshapedTarget = target;

         if (!ShapeOperations::areBroadcastable(prediction.shape(), target.shape())) {
            throw std::invalid_argument("Shapes of prediction and target are not broadcast-compatible.");
        }

        Tensor grad = (2 * (prediction - reshapedTarget) / prediction.shape().size());
        return grad / prediction.shape()[0];
    }
};

} // namespace sdnn

#endif // MSELOSS_HPP

#ifndef CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include "smart_dnn/Loss.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/Shape/ShapeOperations.hpp"
#include <cmath>

namespace sdnn {

class CategoricalCrossEntropyLoss : public Loss {
    static constexpr float epsilon = 1e-7f; // To prevent log(0) or log(1) for numerical stability

public:
    ~CategoricalCrossEntropyLoss() override = default;

    Tensor compute(const Tensor& prediction, const Tensor& target) override {
        if (!ShapeOperations::areBroadcastable(prediction.shape(), target.shape())) {
            throw std::invalid_argument("Shapes of prediction and target are not broadcast-compatible. Mismatch in shapes, prediction: " + prediction.shape().toString() + " and target: " + target.shape().toString());
        } 

        Tensor clippedPred = clip(prediction, epsilon, 1.0f - epsilon);
        Tensor batchLoss = sum(target * log(clippedPred), {1}) * -1;
        return mean(batchLoss);
    }

   Tensor gradient(const Tensor& prediction, const Tensor& target) override {
        if (!ShapeOperations::areBroadcastable(prediction.shape(), target.shape())) {
            throw std::invalid_argument("Shapes of prediction and target are not broadcast-compatible.");
        }

        Tensor expandedTarget = zeros(prediction.shape(), prediction.type());
        if (target.shape() != prediction.shape()) {
            size_t batchSize = prediction.shape()[0];
            for (size_t i = 0; i < batchSize; ++i) {
                size_t targetClass = static_cast<size_t>(target.at<float>(i));
                expandedTarget.set({i, targetClass}, 1.0f);
            }
        } else {
            expandedTarget = target;
        }

        return (prediction - expandedTarget) / prediction.shape()[0];
    }
};


} // namespace sdnn

#endif // CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
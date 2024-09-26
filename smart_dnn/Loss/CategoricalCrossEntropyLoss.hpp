#ifndef CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include "smart_dnn/Loss.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/shape/ShapeOperations.hpp"
#include <cmath>

namespace sdnn {

class CategoricalCrossEntropyLoss : public Loss {
    static constexpr float epsilon = 1e-7f; // To prevent log(0) or log(1) for numerical stability

public:
    ~CategoricalCrossEntropyLoss() override = default;

    Tensor compute(const Tensor& prediction, const Tensor& target) override {
        if (!ShapeOperations::areBroadcastable(prediction.shape(), target.shape())) {
            throw std::invalid_argument("Shapes of prediction and target are not broadcast-compatible.");
        }

        Tensor clippedPred = clip(prediction, epsilon, 1.0f - epsilon);
        Tensor loss = sum(target * log(clippedPred)) * -1;
        
        return loss;
    }

    Tensor gradient(const Tensor& prediction, const Tensor& target) override {
        if (!ShapeOperations::areBroadcastable(prediction.shape(), target.shape())) {
            throw std::invalid_argument("Shapes of prediction and target are not broadcast-compatible.");
        }

        Tensor clippedPred = clip(prediction, epsilon, 1.0f - epsilon);
        
        Tensor expandedTarget = zeros(prediction.shape(), prediction.type());
        size_t y_axis = prediction.shape()[0];
        size_t x_axis = prediction.shape()[1];
        for (size_t i = 0; i < y_axis; ++i) {
            for (size_t j = 0; j < x_axis; ++j) {
                expandedTarget.set({i, j}, target.at<float>(i));
            }
        }
        
        Tensor grad = (clippedPred - expandedTarget) / clippedPred;
        return grad;
    }
};

} // namespace sdnn

#endif // CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
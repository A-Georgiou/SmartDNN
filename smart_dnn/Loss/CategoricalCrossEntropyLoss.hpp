#ifndef CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include "smart_dnn/Loss.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include <cmath>
#include <numeric>

namespace sdnn {

class CategoricalCrossEntropyLoss : public Loss {
    static constexpr auto epsilon = 1e-7; // To prevent log(0) or log(1) for numerical stability

public:

    /*
    
    Input: prediction and target tensors of shape (batchSize, numClasses)
    Output: scalar tensor representing the loss
    
    */

    Tensor compute(const Tensor& prediction, const Tensor& target) override {
        if (prediction.shape() != target.shape()) {
            throw std::invalid_argument("Prediction and target shapes must match");
        }

        int batchSize = prediction.shape()[0];
        int numClasses = prediction.shape()[1];

        float loss = 0;
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                float predValue = std::min(std::max(prediction.at({i, j}), epsilon), 1. - epsilon);
                loss -= target.at({i, j}) * std::log(predValue);
            }
        }

        return Tensor({1}, loss / batchSize);  // Average over batch
    }

    /*
    
    Input: Prediction: 2D tensor (batchSize, numClasses)
           Target: 2D tensor (batchSize, numClasses)
    Output: Gradient: 2D tensor (batchSize, numClasses)
    
    */
    Tensor gradient(const Tensor& prediction, const Tensor& target) override {
        if (prediction.shape() != target.shape()) {
            throw std::invalid_argument("Prediction and target shapes must match");
        }

        int batchSize = prediction.shape()[0];
        int numClasses = prediction.shape()[1];

        Tensor grad(prediction.shape(), 0, prediction.type());

        // Calculate gradients
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                auto predValue = std::min(std::max(prediction.at({i, j}), epsilon), 1. - epsilon);
                grad.at({i, j}) = (predValue - target.at({i, j}));
            }
        }

        return grad / batchSize;
    }
};

} // namespace sdnn

#endif // CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

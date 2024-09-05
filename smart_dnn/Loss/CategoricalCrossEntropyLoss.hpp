#ifndef CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include "../Loss.hpp"
#include "../Tensor/Tensor.hpp"
#include "../TensorOperations.hpp"
#include <cmath>
#include <numeric>

namespace smart_dnn {

template <typename T = float>
class CategoricalCrossEntropyLoss : public Loss<T> {
    using TensorType = Tensor<T>;

    static constexpr T epsilon = T(1e-7); // To prevent log(0) or log(1) for numerical stability

public:

    /*
    
    Input: prediction and target tensors of shape (batchSize, numClasses)
    Output: scalar tensor representing the loss
    
    */
    TensorType compute(const TensorType& prediction, const TensorType& target) override {
        if (prediction.getShape() != target.getShape()) {
            throw std::invalid_argument("Prediction and target shapes must match");
        }

        int batchSize = prediction.getShape()[0];
        int numClasses = prediction.getShape()[1];

        T loss = T(0);
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                T predValue = std::min(std::max(prediction.at({i, j}), epsilon), T(1.0) - epsilon);
                //T predValue = prediction.at({i, j});
                loss -= target.at({i, j}) * std::log(predValue);
            }
        }

        return TensorType({1}, loss / batchSize);  // Average over batch
    }

    /*
    
    Input: Prediction: 2D tensor (batchSize, numClasses)
           Target: 2D tensor (batchSize, numClasses)
    Output: Gradient: 2D tensor (batchSize, numClasses)
    
    */
    TensorType gradient(const TensorType& prediction, const TensorType& target) override {
        if (prediction.getShape() != target.getShape()) {
            throw std::invalid_argument("Prediction and target shapes must match");
        }

        int batchSize = prediction.getShape()[0];
        int numClasses = prediction.getShape()[1];

        TensorType grad(prediction.getShape());

        // Calculate gradients
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                T predValue = std::min(std::max(prediction.at({i, j}), epsilon), T(1.0) - epsilon);
                //T predValue = prediction.at({i, j});
                grad.at({i, j}) = (predValue - target.at({i, j}));
            }
        }

        return grad / batchSize;
    }

    void save(std::ostream& os) const override {
        // No parameters to save for this loss function
    }

    void load(std::istream& is) override {
        // No parameters to load for this loss function
    }
};

} // namespace smart_dnn

#endif // CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#ifndef CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include "../Loss.hpp"
#include "../Tensor/Tensor.hpp"
#include "../TensorOperations.hpp"
#include <cmath>
#include <numeric>

template <typename T=float>
class CategoricalCrossEntropyLoss : public Loss {
    using TensorType = Tensor<T>;
public:
    TensorType compute(const TensorType& prediction, const TensorType& target) override {
        if (prediction.shape() != target.shape()) {
            throw std::invalid_argument("Prediction and target shapes must match, mismatch: " + std::to_string(prediction.shape().size()) + " != " + std::to_string(target.shape().size()));
        }

        const T* predData = prediction.getData();
        const T* targetData = target.getData();
        size_t size = prediction.shape().size();

        T loss = T(0);
        for (size_t i = 0; i < size; ++i) {
            loss -= targetData[i] * std::log(predData[i] + T(1e-7));
        }

        return loss / prediction.shape()[0];  // Normalize by batch size
    }

    TensorType gradient(const TensorType& prediction, const TensorType& target) override {
        if (prediction.shape() != target.shape()) {
            throw std::invalid_argument("Prediction and target shapes must match");
        }

        TensorType grad(prediction.shape());
        T* gradData = grad.getData();
        const T* predData = prediction.getData();
        const T* targetData = target.getData();
        size_t size = prediction.shape().size();

        for (size_t i = 0; i < size; ++i) {
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

#ifndef CATEGORICAL_CROSS_ENTROPY_LOSS_HPP
#define CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

#include "../Loss.hpp"
#include "../Tensor/Tensor.hpp"
#include "../TensorOperations.hpp"
#include <cmath>
#include <numeric>

namespace smart_dnn {

template <typename T=float>
class CategoricalCrossEntropyLoss : public Loss<T> {
    using TensorType = Tensor<T>;
public:
    TensorType compute(const TensorType& prediction, const TensorType& target) override {
        if (prediction.getShape() != target.getShape()) {
            throw std::invalid_argument("Prediction and target shapes must match, mismatch: " + std::to_string(prediction.getShape().size()) + " != " + std::to_string(target.getShape().size()));
        }

        const T* predData = prediction.getData().data();
        const T* targetData = target.getData().data();
        size_t size = prediction.getShape().size();

        T loss = T(0);
        for (size_t i = 0; i < size; ++i) {
            loss -= targetData[i] * std::log(predData[i] + T(1e-7));
        }

        loss /= prediction.getShape()[0];

        return TensorType({1}, loss);  // Normalize by batch size
    }

    TensorType gradient(const TensorType& prediction, const TensorType& target) override {
        if (prediction.getShape() != target.getShape()) {
            throw std::invalid_argument("Prediction and target shapes must match");
        }

        TensorType grad(prediction.getShape());
        T* gradData = grad.getData().data();
        const T* predData = prediction.getData().data();
        const T* targetData = target.getData().data();
        size_t size = prediction.getShape().size();

        for (size_t i = 0; i < size; ++i) {
            gradData[i] = (predData[i] - targetData[i]) / prediction.getShape()[0];
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

} // namespace smart_dnn

#endif // CATEGORICAL_CROSS_ENTROPY_LOSS_HPP

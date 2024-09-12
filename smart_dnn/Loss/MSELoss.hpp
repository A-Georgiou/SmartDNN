#ifndef MSELOSS_HPP
#define MSELOSS_HPP

#include "smart_dnn/Loss.hpp"
#include "smart_dnn/tensor/Tensor.hpp"
#include "smart_dnn/tensor/AdvancedTensorOperations.hpp"
#include <cmath>

namespace smart_dnn {

template <typename T=float>
class MSELoss : public Loss<T> {
    using TensorType = Tensor<T>;
public:
    ~MSELoss() override = default;

    TensorType compute(const TensorType& prediction, const TensorType& target) override {
        TensorType reshapedTarget = target;

        if (prediction.getShape() != target.getShape()) {
            if (target.getShape().rank() == 1 && target.getShape()[0] == prediction.getShape()[0]) {
                reshapedTarget = AdvancedTensorOperations<T>::reshape(target, prediction.getShape());
            } else {
                throw std::invalid_argument("Shapes of prediction and target do not match and cannot be broadcast.");
            }
        }

        TensorType diff = prediction - reshapedTarget;
        TensorType mse = AdvancedTensorOperations<T>::sum((diff * diff)) / diff.getShape().size();
        return mse;
    }

    TensorType gradient(const TensorType& prediction, const TensorType& target) override {
        TensorType reshapedTarget = target;

        if (prediction.getShape() != target.getShape()) {
            if (target.getShape().rank() == 1 && target.getShape()[0] == prediction.getShape()[0]) {
                reshapedTarget = AdvancedTensorOperations<T>::reshape(target, prediction.getShape());
            } else {
                throw std::invalid_argument("Shapes of prediction and target do not match and cannot be broadcast.");
            }
        }

        TensorType grad = T(2) * (prediction - reshapedTarget) / prediction.getShape().size();
        return grad;
    }
};

} // namespace smart_dnn

#endif // MSELOSS_HPP

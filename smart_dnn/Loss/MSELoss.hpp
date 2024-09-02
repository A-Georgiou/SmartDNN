#ifndef MSELOSS_HPP
#define MSELOSS_HPP

#include "../Loss.hpp"
#include "../Tensor/Tensor.hpp"
#include "../Tensor/AdvancedTensorOperations.hpp"
#include <cmath>

namespace smart_dnn {

template <typename T=float>
class MSELoss : public Loss<T> {
public:
    ~MSELoss() override = default;

    Tensor<T> compute(const Tensor<T>& prediction, const Tensor<T>& target) override {
        Tensor<T> reshapedTarget = target;

        if (prediction.getShape() != target.getShape()) {
            if (target.getShape().rank() == 1 && target.getShape()[0] == prediction.getShape()[0]) {
                reshapedTarget = AdvancedTensorOperations<T>::reshape(target, prediction.getShape());
            } else {
                throw std::invalid_argument("Shapes of prediction and target do not match and cannot be broadcast.");
            }
        }

        Tensor<T> diff = prediction - reshapedTarget;
        Tensor<T> mse = AdvancedTensorOperations<T>::sum((diff * diff)) / diff.getShape().size();
        return mse;
    }

    Tensor<T> gradient(const Tensor<T>& prediction, const Tensor<T>& target) override {
        Tensor<T> reshapedTarget = target;

        if (prediction.getShape() != target.getShape()) {
            if (target.getShape().rank() == 1 && target.getShape()[0] == prediction.getShape()[0]) {
                reshapedTarget = AdvancedTensorOperations<T>::reshape(target, prediction.getShape());
            } else {
                throw std::invalid_argument("Shapes of prediction and target do not match and cannot be broadcast.");
            }
        }

        Tensor<T> grad = T(2) * (prediction - reshapedTarget) / prediction.getShape().size();
        return grad;
    }

    void save(std::ostream& os) const override {
        return;
    }

    void load(std::istream& is) override {
        return;
    }
};

} // namespace smart_dnn

#endif // MSELOSS_HPP

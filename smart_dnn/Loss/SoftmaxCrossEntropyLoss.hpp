#ifndef SOFTMAX_CROSS_ENTROPY_LOSS_HPP
#define SOFTMAX_CROSS_ENTROPY_LOSS_HPP

#include "smart_dnn/Loss.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace smart_dnn {

// Cross-entropy on raw logits (..., classes) against class indices (...), skipping rows whose target is ignoreIndex.
template <typename T = float>
class SoftmaxCrossEntropyLoss : public Loss<T> {
public:
    explicit SoftmaxCrossEntropyLoss(int ignoreIndex = -1) : ignoreIndex(ignoreIndex) {}

    Tensor<T> compute(const Tensor<T>& logits, const Tensor<T>& targets) override {
        const size_t classes = validate(logits, targets);
        const T* rowLogits = logits.getData().data();
        T loss = T(0);
        size_t counted = 0;

        for (size_t row = 0; row < targets.getShape().size(); ++row, rowLogits += classes) {
            const int target = static_cast<int>(targets.getData()[row]);
            if (target == ignoreIndex) {
                continue;
            }

            const T maxValue = *std::max_element(rowLogits, rowLogits + classes);
            T sum = T(0);
            for (size_t j = 0; j < classes; ++j) {
                sum += std::exp(rowLogits[j] - maxValue);
            }
            loss += std::log(sum) - (rowLogits[target] - maxValue);
            ++counted;
        }

        return Tensor<T>({1}, counted > 0 ? loss / T(counted) : T(0));
    }

    // Gradient with respect to the logits, (softmax - onehot) / counted rows, so the model needs no Softmax layer.
    Tensor<T> gradient(const Tensor<T>& logits, const Tensor<T>& targets) override {
        const size_t classes = validate(logits, targets);
        Tensor<T> grad(logits.getShape(), T(0));
        const T* rowLogits = logits.getData().data();
        T* rowGrad = grad.getData().data();
        size_t counted = 0;

        for (size_t row = 0; row < targets.getShape().size(); ++row, rowLogits += classes, rowGrad += classes) {
            const int target = static_cast<int>(targets.getData()[row]);
            if (target == ignoreIndex) {
                continue;
            }

            const T maxValue = *std::max_element(rowLogits, rowLogits + classes);
            T sum = T(0);
            for (size_t j = 0; j < classes; ++j) {
                rowGrad[j] = std::exp(rowLogits[j] - maxValue);
                sum += rowGrad[j];
            }
            for (size_t j = 0; j < classes; ++j) {
                rowGrad[j] /= sum;
            }
            rowGrad[target] -= T(1);
            ++counted;
        }

        return counted > 0 ? grad / T(counted) : grad;
    }

    size_t averagedTerms(const Tensor<T>& logits, const Tensor<T>& targets) const override {
        (void)logits;
        const auto& values = targets.getData();
        return static_cast<size_t>(std::count_if(values.begin(), values.end(), [&](T target) { return static_cast<int>(target) != ignoreIndex; }));
    }

private:
    int ignoreIndex;

    // Returns the class count once targets have the logits' shape without the last axis and hold valid indices.
    size_t validate(const Tensor<T>& logits, const Tensor<T>& targets) const {
        std::vector<int> targetDimensions = logits.getShape().getDimensions();
        const int classes = targetDimensions.back();
        targetDimensions.pop_back();
        if (logits.getShape().rank() < 2 || targets.getShape().getDimensions() != targetDimensions) {
            throw std::invalid_argument("Targets must have the logits' shape without the class axis. Got: "
                + logits.getShape().toString() + " and " + targets.getShape().toString());
        }

        for (T value : targets.getData()) {
            const bool isClass = value >= T(0) && value < T(classes) && value == std::floor(value);
            if (!isClass && value != T(ignoreIndex)) {
                throw std::invalid_argument("Targets must be class indices below " + std::to_string(classes) + " or the ignore index");
            }
        }
        return classes;
    }
};

} // namespace smart_dnn

#endif // SOFTMAX_CROSS_ENTROPY_LOSS_HPP

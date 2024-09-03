#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "../Tensor/Tensor.hpp"
#include "../Layer.hpp"
#include "../TensorOperations.hpp"

namespace smart_dnn {

template <typename T>
class DropoutLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    DropoutLayer(T dropoutRate) : dropoutRate(dropoutRate) {}

    TensorType forward(TensorType& input) override {
        if (trainingMode) {
            mask = Tensor::rand(input.getShape());
            mask = (*mask).apply([this](T x) { return x > dropoutRate ? T(1) : T(0); });
            return input * mask * (T(1) / (T(1) - dropoutRate));
        } else {
            return input; // No dropout during inference
        }
    }

    TensorType backward(TensorType& gradOutput) override {
        if (gradOutput.getShape() != mask.value.getShape()) {
            throw std::invalid_argument("Mask not initialised or has wrong shape in Dropout Layer.");
        }
        if (trainingMode) {
            return gradOutput * mask;
        }
        return gradOutput;
    }

    void updateWeights(Optimizer& optimizer) override {}

private:
    T dropoutRate;
    std::optional<TensorType> mask;
};

} // namespace smart_dnn

#endif // DROPOUT_LAYER_HPP
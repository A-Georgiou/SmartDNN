#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "smart_dnn/tensor/Tensor.hpp"
#include "smart_dnn/Layer.hpp"

namespace sdnn {

template <typename T=float>
class DropoutLayer : public Layer<T> {
    using TensorType = Tensor<T>;
public:
    DropoutLayer(T dropoutRate) : dropoutRate(dropoutRate) {}

    TensorType forward(const TensorType& input) override {
        if (this->trainingMode) {
            mask = TensorType::rand(input.getShape());
            mask = (*mask).apply([this](T x) { return x > dropoutRate ? T(1) : T(0); });
            return input * (*mask) * (T(1) / (T(1) - dropoutRate));
        } else {
            return input; // No dropout during inference
        }
    }

    TensorType backward(const TensorType& gradOutput) override {
        if (gradOutput.getShape() != (*mask).getShape()) {
            throw std::invalid_argument("Mask not initialised or has wrong shape in Dropout Layer.");
        }
        if (this->trainingMode) {
            return gradOutput * (*mask);
        }
        return gradOutput;
    }

private:
    T dropoutRate;
    std::optional<TensorType> mask;
};

} // namespace sdnn

#endif // DROPOUT_LAYER_HPP
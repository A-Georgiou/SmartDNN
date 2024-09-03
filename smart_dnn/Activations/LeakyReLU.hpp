#ifndef LEAKY_RELU_HPP
#define LEAKY_RELU_HPP

#include "../Activation.hpp"
#include "../Tensor/Tensor.hpp"
#include "../Tensor/AdvancedTensorOperations.hpp"
#include <algorithm>

namespace smart_dnn {

template <typename T = float>
class LeakyReLU : public Activation<T> {
public:
    explicit LeakyReLU(T alpha = T(0.01)) : alpha(alpha) {}

    Tensor<T> forward(const Tensor<T>& input) const override {
        Tensor<T> output(input);
        return AdvancedTensorOperations<T>::apply(output, [this](T x) { return (x > 0) ? x : (alpha * x); });
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        Tensor<T> gradInput(input.getShape());
        T* gradInputData = gradInput.getData().data();

        const T* inputData = input.getData().data();
        const T* gradOutputData = gradOutput.getData().data();

        int size = input.getShape().size();

        for (int i = 0; i < size; ++i) {
            gradInputData[i] = (inputData[i] > 0) ? gradOutputData[i] : alpha * gradOutputData[i];
        }

        return gradInput;
    }

private:
    T alpha;
};

}

#endif // LEAKY_RELU_HPP

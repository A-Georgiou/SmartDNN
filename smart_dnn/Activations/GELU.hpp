#ifndef GELU_HPP
#define GELU_HPP

#include "smart_dnn/Activation.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include <cmath>
#include <stdexcept>

namespace smart_dnn {

// Gaussian error linear unit, tanh approximation as in GPT-2: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))).
template <typename T = float>
class GELU : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        Tensor<T> output(input.getShape(), T(0));
        const T* x = input.getData().data();
        T* y = output.getData().data();
        for (size_t i = 0; i < input.getShape().size(); ++i) {
            y[i] = T(0.5) * x[i] * (T(1) + std::tanh(inner(x[i])));
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& input, const Tensor<T>& gradOutput) const override {
        if (gradOutput.getShape() != input.getShape()) {
            throw std::invalid_argument("GELU backward needs gradOutput with the input's shape");
        }

        Tensor<T> gradInput(input.getShape(), T(0));
        const T* x = input.getData().data();
        const T* dy = gradOutput.getData().data();
        T* dx = gradInput.getData().data();
        for (size_t i = 0; i < input.getShape().size(); ++i) {
            const T t = std::tanh(inner(x[i]));
            const T innerSlope = scale * (T(1) + T(3) * cubic * x[i] * x[i]);
            dx[i] = dy[i] * (T(0.5) * (T(1) + t) + T(0.5) * x[i] * (T(1) - t * t) * innerSlope);
        }
        return gradInput;
    }

private:
    static constexpr T scale = T(0.7978845608028654);  // sqrt(2 / pi)
    static constexpr T cubic = T(0.044715);

    static T inner(T x) {
        return scale * (x + cubic * x * x * x);
    }
};

} // namespace smart_dnn

#endif // GELU_HPP

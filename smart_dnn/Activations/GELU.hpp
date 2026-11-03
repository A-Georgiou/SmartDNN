#ifndef GELU_HPP
#define GELU_HPP

#include "smart_dnn/Activation.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include <cmath>
#include <stdexcept>

namespace smart_dnn {

// GPT-2's tanh GELU 0.5x(1 + tanh(u)), u = sqrt(2/pi)(x + 0.044715x^3), evaluated as x * sigmoid(2u) from e = exp(-2|u|) <= 1 so tails neither cancel nor overflow.
template <typename T = float>
class GELU : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        Tensor<T> output(input.getShape(), T(0));
        const T* x = input.getData().data();
        T* y = output.getData().data();
        for (size_t i = 0; i < input.getShape().size(); ++i) {
            const T u = inner(x[i]);
            const T e = std::exp(T(-2) * std::abs(u));
            y[i] = x[i] * (u >= T(0) ? T(1) : e) / (T(1) + e);
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
            const T u = inner(x[i]);
            const T e = std::exp(T(-2) * std::abs(u));
            const T d = T(1) / (T(1) + e);
            // The derivative is s + 2x s(1 - s) u' with s(1 - s) = e d^2, whose term is exactly zero once e underflows.
            const T slope = e > T(0) ? T(2) * x[i] * e * d * d * scale * (T(1) + T(3) * cubic * x[i] * x[i]) : T(0);
            dx[i] = dy[i] * ((u >= T(0) ? T(1) : e) * d + slope);
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

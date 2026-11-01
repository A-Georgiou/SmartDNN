#ifndef GRADIENT_CHECK_HPP
#define GRADIENT_CHECK_HPP

#include "smart_dnn/Tensor/Tensor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace smart_dnn {

template <typename T>
struct GradientCheckOptions {
    T epsilon = T(1e-3);
    T tolerance = T(1e-2);
};

template <typename T>
struct GradientCheckResult {
    bool passed = true;
    size_t index = 0;
    T analytic = T(0);
    T numeric = T(0);
    T error = T(0);
};

/*

    Gradient Check
    --------------
    Compares analyticGradient with central differences of objective around point,
    (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon), one coordinate at a time.
    A coordinate passes when |analytic - numeric| <= tolerance * max(1, |analytic|, |numeric|).
    The result describes the coordinate with the largest error, and point is never modified.

    To check a layer's input gradient, use objective(x) = sum(layer.forward(x) * upstream)
    and analyticGradient = layer.backward(upstream) after a forward pass at point.

*/
template <typename T, typename Objective>
GradientCheckResult<T> checkGradient(Objective&& objective, const Tensor<T>& point, const Tensor<T>& analyticGradient,
                                     const GradientCheckOptions<T>& options = {}) {
    if (analyticGradient.getShape() != point.getShape()) {
        throw std::invalid_argument("Analytic gradient shape " + analyticGradient.getShape().toString()
            + " does not match point shape " + point.getShape().toString());
    }
    if (!(options.epsilon > T(0)) || !(options.tolerance >= T(0))) {
        throw std::invalid_argument("Gradient check needs a positive epsilon and a non-negative tolerance");
    }

    Tensor<T> probe = point;
    GradientCheckResult<T> result;
    for (size_t i = 0; i < point.getShape().size(); ++i) {
        const T original = probe[i];
        probe[i] = original + options.epsilon;
        const T plus = objective(probe);
        probe[i] = original - options.epsilon;
        const T minus = objective(probe);
        probe[i] = original;

        const T numeric = (plus - minus) / (T(2) * options.epsilon);
        const T analytic = analyticGradient[i];
        const T error = std::abs(analytic - numeric) / std::max({T(1), std::abs(analytic), std::abs(numeric)});
        // A NaN error is kept as the worst, so non-finite values always fail.
        if (std::isnan(error) || error > result.error) {
            result.index = i;
            result.analytic = analytic;
            result.numeric = numeric;
            result.error = error;
        }
    }

    result.passed = result.error <= options.tolerance;
    return result;
}

} // namespace smart_dnn

#endif // GRADIENT_CHECK_HPP

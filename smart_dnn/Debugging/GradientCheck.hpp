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
    size_t checked = 0;
    T maxAbsoluteError = T(0);
    T maxRelativeError = T(0);  // |analytic - numeric| / max(1, |analytic|, |numeric|), compared with the tolerance.
    size_t index = 0;  // Flat index of the largest relative error, with its two values below.
    T analytic = T(0);
    T numeric = T(0);
};

// Checks analyticGradient against central differences of objective at point; for a layer use sum(forward(x) * upstream) and backward(upstream).
template <typename T, typename Objective>
GradientCheckResult<T> checkGradient(Objective&& objective, const Tensor<T>& point, const Tensor<T>& analyticGradient,
                                     const GradientCheckOptions<T>& options = {}) {
    if (analyticGradient.getShape() != point.getShape()) {
        throw std::invalid_argument("Analytic gradient shape " + analyticGradient.getShape().toString()
            + " does not match point shape " + point.getShape().toString());
    }
    if (!std::isfinite(options.epsilon) || !std::isfinite(options.tolerance) || options.epsilon <= T(0) || options.tolerance < T(0)) {
        throw std::invalid_argument("Gradient check needs a finite positive epsilon and a finite non-negative tolerance");
    }

    Tensor<T> probe = point;
    GradientCheckResult<T> result;
    for (size_t i = 0; i < point.getShape().size(); ++i) {
        const T original = probe[i];
        const T up = original + options.epsilon;
        const T down = original - options.epsilon;
        probe[i] = up;
        const T plus = objective(probe);
        probe[i] = down;
        const T minus = objective(probe);
        probe[i] = original;

        // Divide by the step actually taken; a step lost to rounding gives 0 / 0 and fails.
        const T numeric = (plus - minus) / (up - down);
        const T analytic = analyticGradient[i];
        const T absoluteError = std::abs(analytic - numeric);
        const T relativeError = absoluteError / std::max({T(1), std::abs(analytic), std::abs(numeric)});
        // A NaN error stays the maximum, so non-finite values always fail.
        if (std::isnan(absoluteError) || absoluteError > result.maxAbsoluteError) {
            result.maxAbsoluteError = absoluteError;
        }
        if (std::isnan(relativeError) || relativeError > result.maxRelativeError) {
            result.maxRelativeError = relativeError;
            result.index = i;
            result.analytic = analytic;
            result.numeric = numeric;
        }
        ++result.checked;
    }

    result.passed = result.maxRelativeError <= options.tolerance;
    return result;
}

} // namespace smart_dnn

#endif // GRADIENT_CHECK_HPP

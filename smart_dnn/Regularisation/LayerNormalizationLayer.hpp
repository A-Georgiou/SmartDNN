#ifndef LAYER_NORMALIZATION_LAYER_HPP
#define LAYER_NORMALIZATION_LAYER_HPP

#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace smart_dnn {

// Normalises each row of the last axis to zero mean and unit variance, then applies a learned scale (gamma) and shift (beta).
template <typename T = float>
class LayerNormalizationLayer : public Layer<T> {
public:
    LayerNormalizationLayer(int features, T epsilon = T(1e-5))
        : gamma(Shape({1, features}), T(1)), beta(Shape({1, features}), T(0)),
          gammaGrad(Shape({1, features}), T(0)), betaGrad(Shape({1, features}), T(0)), epsilon(epsilon) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        const Shape& shape = input.getShape();
        const size_t features = gamma.getShape()[1];
        if (static_cast<size_t>(shape[shape.rank() - 1]) != features) {
            throw std::invalid_argument("LayerNormalization expects " + std::to_string(features) + " features in the last axis, got " + shape.toString());
        }

        const size_t rows = shape.size() / features;
        normalized.emplace(shape, T(0));
        inverseDeviations.resize(rows);
        Tensor<T> output(shape, T(0));
        const T* x = input.getData().data();
        T* xHat = normalized->getData().data();
        T* y = output.getData().data();
        const T* g = gamma.getData().data();
        const T* b = beta.getData().data();

        for (size_t r = 0; r < rows; ++r) {
            const size_t offset = r * features;
            T mean = T(0);
            for (size_t j = 0; j < features; ++j) {
                mean += x[offset + j];
            }
            mean /= T(features);
            T variance = T(0);
            for (size_t j = 0; j < features; ++j) {
                variance += (x[offset + j] - mean) * (x[offset + j] - mean);
            }
            inverseDeviations[r] = T(1) / std::sqrt(variance / T(features) + epsilon);
            for (size_t j = 0; j < features; ++j) {
                xHat[offset + j] = (x[offset + j] - mean) * inverseDeviations[r];
                y[offset + j] = g[j] * xHat[offset + j] + b[j];
            }
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        if (!normalized || gradOutput.getShape() != normalized->getShape()) {
            throw std::invalid_argument("LayerNormalization backward needs gradOutput with the last forward pass's shape");
        }

        const size_t features = gamma.getShape()[1];
        const size_t rows = gradOutput.getShape().size() / features;
        Tensor<T> gradInput(gradOutput.getShape(), T(0));
        const T* dy = gradOutput.getData().data();
        const T* xHat = normalized->getData().data();
        const T* g = gamma.getData().data();
        T* dx = gradInput.getData().data();
        T* dGamma = gammaGrad.getData().data();
        T* dBeta = betaGrad.getData().data();
        std::fill(dGamma, dGamma + features, T(0));
        std::fill(dBeta, dBeta + features, T(0));

        for (size_t r = 0; r < rows; ++r) {
            const size_t offset = r * features;
            T meanGrad = T(0);
            T meanGradXHat = T(0);
            for (size_t j = 0; j < features; ++j) {
                const T gradXHat = dy[offset + j] * g[j];
                meanGrad += gradXHat;
                meanGradXHat += gradXHat * xHat[offset + j];
                dGamma[j] += dy[offset + j] * xHat[offset + j];
                dBeta[j] += dy[offset + j];
            }
            meanGrad /= T(features);
            meanGradXHat /= T(features);
            // dx = (dxHat - mean(dxHat) - xHat * mean(dxHat * xHat)) / deviation, with dxHat = dy * gamma.
            for (size_t j = 0; j < features; ++j) {
                dx[offset + j] = inverseDeviations[r] * (dy[offset + j] * g[j] - meanGrad - xHat[offset + j] * meanGradXHat);
            }
        }
        return gradInput;
    }

    std::vector<Parameter<T>> parameters() override {
        return {{gamma, gammaGrad}, {beta, betaGrad}};
    }

    void save(std::ostream& os) const override {
        saveTensor(os, gamma);
        saveTensor(os, beta);
    }

    void load(std::istream& is) override {
        loadTensor(is, gamma);
        loadTensor(is, beta);
    }

private:
    Tensor<T> gamma;
    Tensor<T> beta;
    Tensor<T> gammaGrad;
    Tensor<T> betaGrad;
    T epsilon;
    std::optional<Tensor<T>> normalized;
    std::vector<T> inverseDeviations;
};

} // namespace smart_dnn

#endif // LAYER_NORMALIZATION_LAYER_HPP

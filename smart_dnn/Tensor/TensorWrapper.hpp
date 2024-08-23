#ifndef TENSOR_WRAPPER_HPP
#define TENSOR_WRAPPER_HPP

#include "Tensor.hpp"
#include <stdexcept>
#include <optional>

class TensorWrapper {
public:
    // Default constructor creates an empty wrapper
    TensorWrapper() = default;

    // Constructor from Tensor
    explicit TensorWrapper(const Tensor& t) : tensor(t) {}

    // Copy constructor
    TensorWrapper(const TensorWrapper&) = default;

    // Move constructor
    TensorWrapper(TensorWrapper&&) noexcept = default;

    // Copy assignment operator
    TensorWrapper& operator=(const TensorWrapper&) = default;

    // Move assignment operator
    TensorWrapper& operator=(TensorWrapper&&) noexcept = default;

    TensorWrapper& operator=(const Tensor& t) {
        tensor = t;
        return *this;
    }

    TensorWrapper& operator=(Tensor&& t) {
        tensor = std::move(t);
        return *this;
    }

    Tensor& operator*() {
        return get();
    }

    const Tensor& operator*() const {
        return get();
    }


    // Set the wrapped Tensor
    void set(const Tensor& t) {
        tensor = t;
    }

    // Get the wrapped Tensor (const version)
    const Tensor& get() const {
        if (!tensor.has_value()) {
            throw std::runtime_error("Accessing invalid Tensor");
        }
        return tensor.value();
    }

    // Get the wrapped Tensor (non-const version)
    Tensor& get() {
        if (!tensor.has_value()) {
            throw std::runtime_error("Accessing invalid Tensor");
        }
        return tensor.value();
    }

    // Check if the wrapper contains a valid Tensor
    bool valid() const { return tensor.has_value(); }

    // Reset the wrapper to an invalid state
    void reset() {
        tensor.reset();
    }

    // Overload the -> operator for convenient access to Tensor methods
    const Tensor* operator->() const {
        if (!tensor.has_value()) {
            throw std::runtime_error("Accessing invalid Tensor");
        }
        return &tensor.value();
    }

    Tensor* operator->() {
        if (!tensor.has_value()) {
            throw std::runtime_error("Accessing invalid Tensor");
        }
        return &tensor.value();
    }

private:
    std::optional<Tensor> tensor;
};

#endif // TENSOR_WRAPPER_HPP
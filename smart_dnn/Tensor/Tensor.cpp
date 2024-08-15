#include <Tensor.hpp>

Tensor::Tensor(): _shape(), data({}), d_data(nullptr), onGPU(false) {}

Tensor::Tensor(Shape dimensions): _shape(dimensions), data(dimensions.size(), 0.0f), d_data(nullptr), onGPU(false) {}

Tensor::Tensor(Shape dimensions, float value): _shape(), data(dimensions.size(), value), d_data(nullptr), onGPU(false) {}

Tensor::Tensor(const std::vector<float>& data, const Shape& shape) : _shape(shape), data(data), onGPU(false), d_data(nullptr) {
    if (data.size() != shape.size()) {
        throw std::invalid_argument("Data size does not match tensor shape size.");
    }
}

Tensor::Tensor(const Tensor& other): _shape(other._shape), data(other.data), d_data(nullptr), onGPU(false) {}

Tensor::Tensor(Tensor&& other) noexcept 
    : _shape(other._shape), data(std::move(other.data)), d_data(other.d_data), onGPU(other.onGPU) {
    other.d_data = nullptr;
    other.onGPU = false;
}

Tensor::~Tensor() {
    if (onGPU) {
        freeGPUMemory();
    }
}

void Tensor::swap(Tensor& other) noexcept {
    std::swap(_shape, other._shape);
    std::swap(data, other.data);
    std::swap(d_data, other.d_data);
    std::swap(onGPU, other.onGPU);
}

Tensor& Tensor::operator=(Tensor other) {
    freeGPUMemory();
    swap(other);
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        freeGPUMemory();
        _shape = other._shape;
        data.resize(_shape.size());
        data = std::move(other.data);
        d_data = other.d_data;
        onGPU = other.onGPU;
        
        other.d_data = nullptr;
        other.onGPU = false;
    }
    return *this;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (_shape != other._shape) {
        throw std::invalid_argument("Tensor dimensions mismatch: " + _shape.toString() + " vs " + other._shape.toString());
    }
    for (int i = 0; i < _shape.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (_shape != other._shape) {
        throw std::invalid_argument("Tensor dimensions mismatch: " + _shape.toString() + " vs " + other._shape.toString());
    }
    for (int i = 0; i < _shape.size(); ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (_shape != other._shape) {
        throw std::invalid_argument("Tensor dimensions mismatch: " + _shape.toString() + " vs " + other._shape.toString());
    }
    Tensor result(_shape); // Use the constructor that takes dimensions and a value
    for (int i = 0; i < result._shape.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (_shape != other._shape) {
        throw std::invalid_argument("Tensor dimensions mismatch: " + _shape.toString() + " vs " + other._shape.toString());
    }
    Tensor result(_shape);
    for (int i = 0; i < result._shape.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor result(_shape);
    for (int i = 0; i < result._shape.size(); ++i) {
        result.data[i] = data[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(float scalar) const {
    Tensor result(_shape);
    for (int i = 0; i < result._shape.size(); ++i) {
        result.data[i] = data[i] - scalar;
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(_shape);
    for (int i = 0; i < result._shape.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}
#include "../Tensor.hpp"
#include "../TensorOperations.hpp"
#include "../Debugging/Logger.hpp"
#include <utility>

/*

INITIALISATION CONSTRUCTORS

*/

Tensor::Tensor() : _shape(), data(), d_data(nullptr), onGPU(false) {}

Tensor::Tensor(Shape otherShape)
    : _shape(std::move(otherShape)), data(_shape.size(), 0.0f), d_data(nullptr), onGPU(false) {}

Tensor::Tensor(Shape otherShape, float value)
    : _shape(std::move(otherShape)), data(_shape.size(), value), d_data(nullptr), onGPU(false) {}

Tensor::Tensor(Shape otherShape, std::vector<float> data)
    : _shape(std::move(otherShape)), data(std::move(data)), d_data(nullptr), onGPU(false) {
    if (this->data.size() != _shape.size()) {
        throw std::invalid_argument("Data size does not match tensor shape size. Data size: " +
                                    std::to_string(this->data.size()) + ", Shape size: " +
                                    std::to_string(_shape.size()));
    }
}

float& Tensor::operator()(std::initializer_list<int> indices) {
        return data[TensorOperations::flattenIndex(indices, _shape)];
    }

const float& Tensor::operator()(std::initializer_list<int> indices) const {
        return data[TensorOperations::flattenIndex(indices, _shape)];
    }

std::vector<int> Tensor::size() const {
    return _shape.dimensions;
}

int Tensor::size(int axis) const{
    if (axis < 0 || axis >= _shape.rank()) {
        throw std::out_of_range("Axis out of bounds, max rank: " + std::to_string(_shape.rank()));
    }
    return _shape.dimensions[axis];
}

/*

COPY AND MOVE OPERATORS OVERLOADING

*/

void Tensor::swap(Tensor& other) noexcept {
    std::swap(_shape, other._shape);
    std::swap(data, other.data);
    std::swap(d_data, other.d_data);
    std::swap(onGPU, other.onGPU);
}

/*

LOGIC OPERATORS OVERLOADING

*/

Tensor& Tensor::operator+=(const Tensor& other) {
    applyElementWiseOperation(other, std::plus<float>(), this);
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    applyElementWiseOperation(other, std::minus<float>(), this);
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    applyElementWiseOperation(other, std::multiplies<float>(), this);
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    applyElementWiseOperation(other, std::divides<float>(), this);
    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const {
    Tensor result;
    applyElementWiseOperation(other, std::plus<float>(), &result);
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    Tensor result;
    applyElementWiseOperation(other, std::minus<float>(), &result);
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    Tensor result;
    applyElementWiseOperation(other, std::multiplies<float>(), &result);
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    Tensor result;
    applyElementWiseOperation(other, std::divides<float>(), &result);
    return result;
}


Tensor Tensor::operator+(float scalar) const{
    Tensor result(_shape);
    for (int i = 0; i < result._shape.size(); ++i) {
        result.data[i] = data[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(float scalar) const{
    Tensor result(_shape);
    for (int i = 0; i < result._shape.size(); ++i) {
        result.data[i] = data[i] - scalar;
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const{
    Tensor result(_shape);
    for (int i = 0; i < result._shape.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator/(float scalar) const{
    Tensor result(_shape);
    for (int i = 0; i < result._shape.size(); ++i) {
        result.data[i] = data[i] / scalar;
    }
    return result;
}

/*

UTILITY FUNCTIONS

*/

void Tensor::fill(float value) {
    for (int i = 0; i < _shape.size(); ++i) {
        data[i] = value;
    }
}

void Tensor::randomize(float min, float max) {
    for (auto& value : data) {
        value = RandomEngine::getRandRange(min, max);
    }
}

void Tensor::print() const {
    std::cout << "Tensor: " << _shape << std::endl;
}

/*

TENSOR MATHEMATIC FUNCTIONS

*/

void Tensor::add(const Tensor& other){
    *this += other;
}

void Tensor::subtract(const Tensor& other){
    *this -= other;
}

Tensor Tensor::sqrt() const{
    Tensor result(_shape);
    for (int i = 0; i < _shape.size(); ++i) {
        result.data[i] = std::sqrt(data[i]);
    }
    return result;
}

float Tensor::sum() const {
    return std::accumulate(data.begin(), data.end(), 0);
}

Tensor Tensor::sum(int axis) const {
    if (axis < 0 || axis >= _shape.rank()) {
        throw std::out_of_range("Axis out of bounds");
    }

    std::vector<int> newShape = _shape.dimensions;
    newShape.erase(newShape.begin() + axis);

    Tensor result(Shape(newShape), 0.0f);

    for (int i = 0; i < _shape.size(); ++i) {
        std::vector<int> indices = TensorOperations::getIndices(i, _shape);
        indices.erase(indices.begin() + axis);
        int flatIdx = TensorOperations::flattenIndex(indices, Shape(newShape));
        result.data[flatIdx] += data[i];
    }

    return result;
}
void Tensor::transpose(int dim1, int dim2) {
    if (dim1 < 0 || dim1 >= _shape.rank() || dim2 < 0 || dim2 >= _shape.rank()) {
        throw std::out_of_range("Dimensions out of bounds: (" + std::to_string(dim1) + ", " + std::to_string(dim2) + ") for shape with rank: " + std::to_string(_shape.rank()));
    }

    if (dim1 == dim2) return;

    std::vector<int> oldShape = _shape.dimensions;
    std::swap(oldShape[dim1], oldShape[dim2]);

    std::vector<float> newData(data.size());

    for (int i = 0; i < _shape.size(); ++i) {
        std::vector<int> indices = TensorOperations::getIndices(i, _shape);
        std::swap(indices[dim1], indices[dim2]);
        int newIndex = TensorOperations::flattenIndex(indices, Shape(oldShape));
        newData[newIndex] = data[i];
    }

    _shape.dimensions = oldShape;
    data = std::move(newData);
}


void Tensor::reshape(const Shape& newShape){
    if (newShape.dimensions.size() == 1 && newShape.dimensions[0] == -1){
        _shape.dimensions = {_shape.size()};
        return;
    }

    if (newShape.size() != _shape.size()) {
        throw std::invalid_argument("New dimensions must have the same size as the current tensor: " +
        std::to_string(_shape.size()) + " != " + std::to_string(newShape.size()));
    }

    for (int num : newShape.dimensions) {
        if (num <= 0)
            throw std::invalid_argument("New dimensions must be positive");
    }

    _shape = newShape;
}

Tensor Tensor::reshape(const Shape& newShape) const {
    if (newShape.size() != _shape.size()) {
        throw std::invalid_argument("New dimensions must have the same size as the current tensor: " +
        std::to_string(_shape.size()) + " != " + std::to_string(newShape.size()));
    }

    Tensor reshapedTensor(newShape, this->data);
    return reshapedTensor;
}

Tensor Tensor::apply(std::function<float(float)> op) const {
        Tensor result(_shape, this->data);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = op(data[i]); 
        }
        return result; 
    }

/*

CPU / GPU MEMORY MANAGEMENT

*/

void Tensor::toGPU(){
    if (!onGPU) {
        onGPU = true;
    }
}

void Tensor::toCPU(){
    if (onGPU) {
        onGPU = false;
    }
}

bool Tensor::isOnGPU() const {
    return onGPU;
}


void Tensor::allocateGPUMemory(){
    if (!onGPU) {
        onGPU = true;
    }
}

void Tensor::freeGPUMemory(){
    if (onGPU) {
        onGPU = false;
    }
}

void Tensor::copyToGPU(){
    if (!onGPU) {
        allocateGPUMemory();
    }
}

void Tensor::copyToCPU(){
    if (onGPU) {
        onGPU = false;
    }
}

/*

ADDITIONAL ELEMENT WISE OPERATIONS

*/

void Tensor::applyElementWiseOperation(const Tensor& other, std::function<float(float, float)> op, Tensor* result) const {
    checkCompatibility(other);
    Shape resultShape{getBroadcastShape(other)};
    bool inPlace = (result == this);

    if (!inPlace) {
        result->_shape = Shape(resultShape);
        result->data.resize(result->_shape.size());
    }

    // Precompute strides for flattening indices
    std::vector<int> strides1(_shape.rank());
    std::vector<int> strides2(other.shape().rank());
    std::vector<int> stridesResult(resultShape.rank());

    strides1.back() = 1;
    for (int i = _shape.rank() - 2; i >= 0; --i) {
        strides1[i] = strides1[i + 1] * _shape[i + 1];
    }

    strides2.back() = 1;
    for (int i = other.shape().rank() - 2; i >= 0; --i) {
        strides2[i] = strides2[i + 1] * other.shape()[i + 1];
    }

    stridesResult.back() = 1;
    for (int i = resultShape.rank() - 2; i >= 0; --i) {
        stridesResult[i] = stridesResult[i + 1] * resultShape[i + 1];
    }

    // Apply element-wise operation using strides
    std::vector<int> indices(resultShape.rank(), 0);

    for (int i = 0; i < resultShape.size(); ++i) {
        int flatIdx1 = 0;
        int flatIdx2 = 0;
        int flatResultIdx = 0;

        for (int j = 0; j < resultShape.rank(); ++j) {
            flatResultIdx += indices[j] * stridesResult[j];

            if (_shape.rank() > j) {
                if (_shape[j] != 1) {
                    flatIdx1 += indices[j] * strides1[j];
                }
            }

            if (other.shape().rank() > j) {
                if (other.shape()[j] != 1) {
                    flatIdx2 += indices[j] * strides2[j];
                }
            }
        }

        result->data[flatResultIdx] = op(data[flatIdx1], other.data[flatIdx2]);

        // Increment indices
        for (int j = resultShape.rank() - 1; j >= 0; --j) {
            if (++indices[j] < resultShape[j]) {
                break;
            }
            indices[j] = 0;
        }
    }
}


std::vector<int> Tensor::getBroadcastShape(const Tensor& other) const {
    return getBroadcastShape(other.shape());
}

std::vector<int> Tensor::getBroadcastShape(const Shape& newShape) const {
    std::vector<int> shape1 = _shape.dimensions;
    std::vector<int> shape2 = newShape.dimensions;
    std::vector<int> resultShape;

    std::reverse(shape1.begin(), shape1.end());
    std::reverse(shape2.begin(), shape2.end());

    int maxDim = std::max(shape1.size(), shape2.size());
    shape1.resize(maxDim, 1); 
    shape2.resize(maxDim, 1);

    for (int i = 0; i < maxDim; ++i) {
        if (shape1[i] == shape2[i] || shape1[i] == 1 || shape2[i] == 1) {
            resultShape.push_back(std::max(shape1[i], shape2[i]));
        } else {
            throw std::invalid_argument("Tensor dimensions do not match for broadcasting");
        }
    }

    std::reverse(resultShape.begin(), resultShape.end());
    return resultShape;
}

void Tensor::checkCompatibility(const Tensor& other) const {
    getBroadcastShape(other);
}
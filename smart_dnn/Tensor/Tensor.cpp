#include "../Tensor.hpp"
#include "../TensorOperations.hpp"
#include "../Debugging/Logger.hpp"

/*

INITIALISATION CONSTRUCTORS

*/

Tensor::Tensor(): _shape(), data({}), d_data(nullptr), onGPU(false) {}
Tensor::Tensor(Shape dimensions): _shape(dimensions), data(dimensions.size(), 0.0f), d_data(nullptr), onGPU(false) {}
Tensor::Tensor(Shape dimensions, float value): _shape(dimensions), data(dimensions.size(), value), d_data(nullptr), onGPU(false) {}
Tensor::Tensor(const Tensor& other): _shape(other._shape), data(other.data), d_data(nullptr), onGPU(false) {}
Tensor::Tensor(const std::vector<float>& data, const Shape& shape) : _shape(shape), data(data), onGPU(false), d_data(nullptr) {
    if (data.size() != shape.size()) {
        throw std::invalid_argument("Data size does not match tensor shape size.");
    }
}

Tensor::Tensor(Tensor&& other) noexcept 
    : _shape(other._shape), data(std::move(other.data)), d_data(other.d_data), onGPU(other.onGPU) {
    other.d_data = nullptr;
    other.onGPU = false;
}

float& Tensor::operator()(std::initializer_list<int> indices) {
        return data[TensorOperations::flattenIndex(indices, _shape)];
    }

const float& Tensor::operator()(std::initializer_list<int> indices) const {
        return data[TensorOperations::flattenIndex(indices, _shape)];
    }

Tensor::~Tensor() {
    if (onGPU) {
        freeGPUMemory();
    }
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

// Element-wise addition with broadcasting
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

Tensor Tensor::operator/(float scalar) const {
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
    std::cout << "Tensor: " << _shape.toString() << std::endl;
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

int Tensor::sum() const {
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
        throw std::out_of_range("Dimensions out of bounds");
    }

    if (dim1 == dim2) return; 

    std::swap(_shape.dimensions[dim1], _shape.dimensions[dim2]);
    std::vector<bool> visited(_shape.size(), false);

    for (int i = 0; i < _shape.size(); ++i) {
        if (visited[i]) continue;

        int current = i;
        do {
            visited[current] = true;
            std::vector<int> indices = TensorOperations::getIndices(current, _shape);
            std::swap(indices[dim1], indices[dim2]);
            int next = TensorOperations::flattenIndex(indices, _shape);

            if (next != i) {
                std::swap(data[current], data[next]);
            }

            current = next;
        } while (current != i);
    }
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
    std::vector<int> resultShape = getBroadcastShape(other);
    bool inPlace = (result == this);

    if (!inPlace) {
        result->_shape = Shape(resultShape);
        result->data.resize(result->_shape.size());
    }

    int totalElements = std::accumulate(resultShape.begin(), resultShape.end(), 1, std::multiplies<int>());
    std::vector<int> resultIndices(resultShape.size(), 0);

    for (int i = 0; i < totalElements; ++i) {
        int temp = TensorOperations::flattenIndex(resultIndices, Shape(resultShape));

        std::vector<int> idx1(_shape.rank(), 0);
        std::vector<int> idx2(other.shape().rank(), 0);
        for (int j = 0; j < resultShape.size(); ++j) {
            if (_shape.rank() > j)
                idx1[_shape.rank() - j - 1] = (_shape.dimensions[_shape.rank() - j - 1] == 1) ? 0 : resultIndices[resultShape.size() - j - 1];
            if (other.shape().rank() > j)
                idx2[other.shape().rank() - j - 1] = (other.shape().dimensions[other.shape().rank() - j - 1] == 1) ? 0 : resultIndices[resultShape.size() - j - 1];
        }

        int flatIdx1 = TensorOperations::flattenIndex(idx1, _shape);
        int flatIdx2 = TensorOperations::flattenIndex(idx2, other.shape());
        int flatResultIdx = TensorOperations::flattenIndex(resultIndices, Shape(resultShape));

        result->data[flatResultIdx] = op(data[flatIdx1], other.data[flatIdx2]);
    }
}

std::vector<int> Tensor::getBroadcastShape(const Tensor& other) const {
    std::vector<int> shape1 = _shape.dimensions;
    std::vector<int> shape2 = other._shape.dimensions;
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
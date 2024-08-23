#include "../Tensor.hpp"
#include "../TensorOperations.hpp"
#include <utility>
#include <execution>

/*

INITIALISATION CONSTRUCTORS

*/

Tensor::Tensor(Shape otherShape) noexcept
    : _shape(std::move(otherShape)), data(new float[_shape.size()]{}), d_data(nullptr), onGPU(false) {}

Tensor::Tensor(Shape otherShape, float value) noexcept
    : _shape(std::move(otherShape)), data(new float[_shape.size()]), d_data(nullptr), onGPU(false) {
    std::fill_n(data.get(), _shape.size(), value);
}

Tensor::Tensor(Shape otherShape, const float* inputData)
    : _shape(std::move(otherShape)), data(new float[_shape.size()]), d_data(nullptr), onGPU(false) {
    std::copy_n(inputData, _shape.size(), data.get());
}

Tensor::Tensor(const Tensor& other)
    : _shape(other._shape), data(new float[other._shape.size()]), d_data(nullptr), onGPU(false) {
    std::copy_n(other.data.get(), _shape.size(), data.get());
}

Tensor::Tensor(Tensor&& other) noexcept
    : _shape(std::move(other._shape)), data(std::move(other.data)), d_data(other.d_data), onGPU(other.onGPU) {
    other.d_data = nullptr;
    other.onGPU = false;
}

Tensor::~Tensor() {
    if (d_data) {
        freeGPUMemory();
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        _shape = other._shape;
        data.reset(new float[_shape.size()]);
        std::copy_n(other.data.get(), _shape.size(), data.get());
        if (d_data) {
            freeGPUMemory();
        }
        d_data = nullptr;
        onGPU = false;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        _shape = std::move(other._shape);
        data = std::move(other.data);
        d_data = other.d_data;
        onGPU = other.onGPU;
        other.d_data = nullptr;
        other.onGPU = false;
    }
    return *this;
}

float& Tensor::operator()(std::initializer_list<int> indices) {
    return data[TensorOperations::flattenIndex(indices, _shape)];
}

const float& Tensor::operator()(std::initializer_list<int> indices) const {
    return data[TensorOperations::flattenIndex(indices, _shape)];
}

inline std::vector<int> Tensor::size() const noexcept {
    return _shape.getDimensions();
}

inline int Tensor::size(int axis) const {
    if (axis < 0 || axis >= _shape.rank()) {
        throw std::out_of_range("Axis out of bounds, max rank: " + std::to_string(_shape.rank()));
    }
    return _shape[axis];
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor Shape: " << tensor.shape() << std::endl;
    int nDims = tensor.shape().rank();
    std::vector<int> indices(nDims, 0);
    const float* data = tensor.getData();
    int size = tensor.shape().size();
    std::stack<int> bracketStack;

    for (int i = 0; i < size; ++i) {
        os << data[i] << " ";
        indices[nDims - 1] += 1;
        for (int j = nDims - 1; j >= 0; --j) {
            if (indices[j] == (tensor.shape())[j]) {
                indices[j] = 0;
                if (j > 0) {
                    os << std::endl;
                    indices[j - 1] += 1;
                }
            }
        }
    }
    os << std::endl;
    return os;
}

std::string Tensor::toString() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
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

Tensor& Tensor::operator+=(float scalar) noexcept {
    std::transform(data.get(), data.get() + _shape.size(), data.get(), [scalar](float x) { return x + scalar;});
    return *this;
}

Tensor& Tensor::operator-=(float scalar) noexcept {
    std::transform(data.get(), data.get() + _shape.size(), data.get(), [scalar](float x) { return x - scalar;});
    return *this;
}

Tensor& Tensor::operator*=(float scalar) noexcept {
    std::transform(data.get(), data.get() + _shape.size(), data.get(), [scalar](float x) { return x * scalar;});
    return *this;
}

Tensor& Tensor::operator/=(float scalar) noexcept {
    std::transform(data.get(), data.get() + _shape.size(), data.get(), [scalar](float x) { return x / scalar;});
    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const {
    Shape resultShape(getBroadcastShape(other));
    Tensor result(resultShape);
    applyElementWiseOperation(other, std::plus<float>(), &result);
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    Shape resultShape(getBroadcastShape(other));
    Tensor result(resultShape);
    applyElementWiseOperation(other, std::minus<float>(), &result);
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    Shape resultShape(getBroadcastShape(other));
    Tensor result(resultShape);
    applyElementWiseOperation(other, std::multiplies<float>(), &result);
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    Shape resultShape(getBroadcastShape(other));
    Tensor result(resultShape);
    applyElementWiseOperation(other, std::divides<float>(), &result);
    return result;
}

Tensor Tensor::operator+(float scalar) const noexcept {
    Tensor result(_shape, data.get());
    return result += scalar;
}

Tensor Tensor::operator-(float scalar) const noexcept {
    Tensor result(_shape, data.get());
    return result -= scalar;
}

Tensor Tensor::operator*(float scalar) const noexcept {
    Tensor result(_shape);
    return result *= scalar;
}

Tensor Tensor::operator/(float scalar) const noexcept {
    Tensor result(_shape);
    return result /= scalar;
}

/*

UTILITY FUNCTIONS

*/

void Tensor::fill(float value) noexcept {
    std::fill_n(data.get(), _shape.size(), value);
}

void Tensor::randomize(float min, float max) {
    for (int i = 0; i < _shape.size(); ++i) {
        data[i] = RandomEngine::getRandRange(min, max);
    }
}

void Tensor::print() const noexcept {
    std::cout << "Tensor: " << _shape << std::endl;
}

/*

TENSOR MATHEMATIC FUNCTIONS

*/

Tensor Tensor::sqrt() const {
    Tensor result(_shape);
    for (int i = 0; i < _shape.size(); ++i) {
        result.data[i] = std::sqrt(data[i]);
    }
    return result;
}

Tensor Tensor::mean(int axis) const {
    if (axis < 0 || axis >= _shape.rank()) {
        throw std::out_of_range("Axis out of bounds");
    }

    std::vector<int> newShape = _shape.getDimensions();
    newShape.erase(newShape.begin() + axis);

    Tensor result(Shape(newShape), 0.0f);

    for (int i = 0; i < _shape.size(); ++i) {
        std::vector<int> indices = TensorOperations::getIndices(i, _shape);
        indices.erase(indices.begin() + axis);
        int flatIdx = TensorOperations::flattenIndex(indices, Shape(newShape));
        result.data[flatIdx] += data[i];
    }

    result /= _shape[axis];
    return result;
}

Tensor Tensor::var(int axis) const {
    if (axis < 0 || axis >= _shape.rank()) {
        throw std::out_of_range("Axis out of bounds");
    }

    Tensor mean = this->mean(axis);
    Tensor squared = *this - mean;
    squared *= squared;
    return squared.mean(axis);
}

float Tensor::sum() const {
    return std::accumulate(data.get(), data.get() + _shape.size(), 0.0f);
}

Tensor Tensor::sum(int axis) const {
    if (axis < 0 || axis >= _shape.rank()) {
        throw std::out_of_range("Axis out of bounds");
    }

    std::vector<int> newShape = _shape.getDimensions();
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

    std::vector<int> oldShape = _shape.getDimensions();
    std::swap(oldShape[dim1], oldShape[dim2]);

    std::unique_ptr<float[]> newData(new float[_shape.size()]);

    #pragma omp parallel for
    for (int i = 0; i < _shape.size(); ++i) {
        std::vector<int> indices = TensorOperations::getIndices(i, _shape);
        std::swap(indices[dim1], indices[dim2]);
        int newIndex = TensorOperations::flattenIndex(indices, Shape(oldShape));
        newData[newIndex] = data[i];
    }

    _shape.setDimensions(oldShape);
    data = std::move(newData);
}

void Tensor::reshape(const Shape& newShape) {
    if (newShape.getDimensions().size() == 1 && newShape[0] == -1) {
        _shape = {_shape.size()};
        return;
    }

    if (newShape.size() != _shape.size()) {
        throw std::invalid_argument("New dimensions must have the same size as the current tensor: " +
        std::to_string(_shape.size()) + " != " + std::to_string(newShape.size()));
    }

    for (int num : newShape.getDimensions()) {
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

    return Tensor(std::move(newShape), data.get());
}

Tensor Tensor::apply(std::function<float(float)> op) const {
    Tensor result(_shape);
    if (_shape.size() > MIN_PARALLEL_SIZE) {
        #pragma omp parallel for
        for (int i = 0; i < _shape.size(); ++i) {
            result.data[i] = op(data[i]);
        }
    } else {
        for (int i = 0; i < _shape.size(); ++i) {
            result.data[i] = op(data[i]);
        }
    }

    return result;
}

/*

CPU / GPU MEMORY MANAGEMENT

*/

void Tensor::toGPU() {
    if (!onGPU) {
        onGPU = true;
    }
}

void Tensor::toCPU() {
    if (onGPU) {
        onGPU = false;
    }
}

bool Tensor::isOnGPU() const {
    return onGPU;
}

void Tensor::allocateGPUMemory() {
    if (!onGPU) {
        onGPU = true;
    }
}

void Tensor::freeGPUMemory() {
    if (onGPU) {
        onGPU = false;
    }
}

void Tensor::copyToGPU() {
    if (!onGPU) {
        allocateGPUMemory();
    }
}

void Tensor::copyToCPU() {
    if (onGPU) {
        onGPU = false;
    }
}

/*

ADDITIONAL ELEMENT WISE OPERATIONS

*/
inline std::vector<int> computeStrides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

Tensor Tensor::applyElementWiseOperation(const Tensor& other, std::function<float(float, float)> op) const {
    Tensor result(Shape(getBroadcastShape(other)));
    applyElementWiseOperation(other, op, &result);
    return result;
}

void BlasOperationOnTensor(const Tensor& tensor, const Tensor& other, Tensor* result, const std::function<float(float, float)>& op) {
    if (tensor.shape() != other.shape() || tensor.shape() != result->shape()) {
        throw std::invalid_argument("All tensors must have the same shape for element-wise operations.");
    }

    size_t size = tensor.shape().size();
    const float* a = tensor.getData();
    const float* b = other.getData();
    float* c = result->getData();
    std::transform(a, a + size, b, c, op);
}

void Tensor::applyElementWiseOperation(const Tensor& other, std::function<float(float, float)> op, Tensor* result) const {
    if (result->shape() == other.shape() && result->shape() == _shape) {
        BlasOperationOnTensor(*this, other, result, op);
        return;
    }
    
    Shape resultShape{getBroadcastShape(other)};

    auto strides1 = computeStrides(_shape.getDimensions());
    auto strides2 = computeStrides(other.shape().getDimensions());
    auto stridesResult = computeStrides(result->shape().getDimensions());

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

inline std::vector<int> Tensor::getBroadcastShape(const Shape& newShape) const {
    std::vector<int> shape1 = _shape.getDimensions();
    std::vector<int> shape2 = newShape.getDimensions();
    std::vector<int> resultShape;

    std::reverse(shape1.begin(), shape1.end());
    std::reverse(shape2.begin(), shape2.end());

    int maxDim = std::max(shape1.size(), shape2.size());
    shape1.resize(maxDim, 1); 
    shape2.resize(maxDim, 1);

    for (int i = 0; i < maxDim; ++i) {
        if (shape1[i] == shape2[i] || shape1[i] == 1 || shape2[i] == 1) {
            resultShape.emplace_back(std::max(shape1[i], shape2[i]));
        } else {
            throw std::invalid_argument("Tensor dimensions do not match for broadcasting");
        }
    }

    std::reverse(resultShape.begin(), resultShape.end());
    return resultShape;
}

Tensor operator+(float scalar, const Tensor& tensor) {
    return tensor + scalar; 
}

Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

Tensor operator-(float scalar, const Tensor& tensor) noexcept {
    Tensor result(tensor.shape());
    float* resultData = result.getData();
    const float* tensorData = tensor.getData();
    int size = tensor.shape().size();

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        resultData[i] = scalar - tensorData[i];
    }

    return result;
}

Tensor operator/(float scalar, const Tensor& tensor) noexcept {
    Tensor result(tensor.shape());
    float* resultData = result.getData();
    const float* tensorData = tensor.getData();
    int size = tensor.shape().size();

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        resultData[i] = scalar / tensorData[i];
    }

    return result;
}


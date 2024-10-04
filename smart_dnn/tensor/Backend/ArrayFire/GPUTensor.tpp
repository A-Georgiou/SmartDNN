
namespace sdnn {

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, const std::vector<T>& data)
        : shape_(shape), type_(dtype_trait<T>::value) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        data_ = std::make_shared<af::array>(af::dim4(shape.size()), data.data());
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, const T* data, size_t num_elements)
        : shape_(shape), type_(dtype_trait<T>::value) {
        data_ = std::make_shared<af::array>(af::dim4(shape.size()), data, num_elements);
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, T data, dtype type)
        : shape_(shape), type_(type) {
        data_ = std::make_shared<af::array>(af::dim4(shape.size()), data);
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, T data)
        : shape_(shape), type_(dtype_trait<T>::value) {
        data_ = std::make_shared<af::array>(af::dim4(shape.size()), data);
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, std::initializer_list<T> values, dtype type)
        : shape_(shape), type_(type) {
        data_ = std::make_shared<af::array>(af::dim4(shape.size()), values.begin(), values.end());
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, std::initializer_list<T> values)
        : shape_(shape), type_(dtype_trait<T>::value) {
        data_ = std::make_shared<af::array>(af::dim4(shape.size()), values.begin(), values.end());
    }

}
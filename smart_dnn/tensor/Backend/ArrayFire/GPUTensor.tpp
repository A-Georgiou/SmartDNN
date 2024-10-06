
namespace sdnn {
    
    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, const std::vector<T>& data)
        : shape_(shape), type_(dtype_trait<T>::value) {
        if (shape.size() != data.size()) {  // Ensure total size matches data size
            throw std::invalid_argument("Data size does not match shape");
        }
        data_ = std::make_shared<af::array>(utils::shapeToAfDim(shape), data.data(), afHost);
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, const std::vector<T>& data, dtype type)
        : shape_(shape), type_(type) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        data_ = std::make_shared<af::array>(utils::shapeToAfDim(shape), data.data(), afHost);
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, const T* data, size_t num_elements)
        : shape_(shape), type_(dtype_trait<T>::value) {
        if (shape.size() != num_elements) {
            throw std::invalid_argument("Number of elements does not match shape");
        }
        data_ = std::make_shared<af::array>(utils::shapeToAfDim(shape), data, afHost);
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, T value, dtype type)
        : shape_(shape), type_(type) {
        if constexpr (std::is_same_v<T, af::array::array_proxy>) {
            data_ = std::make_shared<af::array>(value);
            if (data_->dims() != utils::shapeToAfDim(shape)) {
                throw std::invalid_argument("Array proxy shape does not match the provided shape");
            }
        } else {
            data_ = std::make_shared<af::array>(af::constant(value, utils::shapeToAfDim(shape), utils::sdnnToAfType(type)));
        }
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, T value)
        : shape_(shape), type_(dtype_trait<T>::value) {
        GPUTensor(shape, value, dtype_trait<T>::value);
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, std::initializer_list<T> values, dtype type)
        : shape_(shape), type_(type) {
        if (shape.size() != values.size()) {
            throw std::invalid_argument("Initializer list size does not match shape");
        }
        data_ = std::make_shared<af::array>(utils::shapeToAfDim(shape), values.begin(), afHost);
    }

}

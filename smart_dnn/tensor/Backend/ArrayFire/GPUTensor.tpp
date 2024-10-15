
namespace sdnn {
    
    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, const std::vector<T>& data)
        : shape_(shape), type_(dtype_trait<T>::value) {
        if (shape.size() != data.size()) {  // Ensure total size matches data size
            throw std::invalid_argument("Data size does not match shape");
        }

        std::vector<T> data_col_major(data.size());
        convertRowMajorToColumnMajor(data.data(), data_col_major.data(), shape);
        data_ = std::make_shared<af::array>(utils::shapeToAfDim(shape), data_col_major.data());
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, const std::vector<T>& data, dtype type)
        : shape_(shape), type_(type) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }

        std::vector<T> data_col_major(data.size());
        convertRowMajorToColumnMajor(data.data(), data_col_major.data(), shape);
        data_ = std::make_shared<af::array>(utils::shapeToAfDim(shape), data_col_major.data());
    }

    template <typename T>
    GPUTensor::GPUTensor(const Shape& shape, const T* data, size_t num_elements)
        : shape_(shape), type_(dtype_trait<T>::value) {
        if (shape.size() != num_elements) {
            throw std::invalid_argument("Number of elements does not match shape");
        }

        std::vector<T> data_col_major(num_elements);
        convertRowMajorToColumnMajor(data, data_col_major.data(), shape);
        data_ = std::make_shared<af::array>(utils::shapeToAfDim(shape), data_col_major.data());
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
            af::array temp = af::constant(value, utils::shapeToAfDim(shape), utils::sdnnToAfType(type));
            data_ = std::make_shared<af::array>(temp);
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
        std::vector<T> values_col_major(values);
        convertRowMajorToColumnMajor(values.begin(), values_col_major.data(), shape);
        data_ = std::make_shared<af::array>(utils::shapeToAfDim(shape), values_col_major.data());
    }

    template <typename T>
    void convertRowMajorToColumnMajor(const T* srcData, T* dstData, const Shape& shape) {
        size_t totalElements = shape.size();
        std::vector<size_t> indices(shape.rank());

        for (size_t idx = 0; idx < totalElements; ++idx) {
            size_t remaining = idx;
            for (int i = shape.rank() - 1; i >= 0; --i) {
                indices[i] = remaining % shape[i];
                remaining /= shape[i];
            }

            size_t colMajorIndex = 0;
            size_t stride = 1;
            for (size_t i = 0; i < shape.rank(); ++i) {
                colMajorIndex += indices[i] * stride;
                stride *= shape[i];
            }

            dstData[colMajorIndex] = srcData[idx];
        }
    }

}

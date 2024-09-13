

namespace sdnn {
    template <typename T>
    Tensor::Tensor(const Shape& shape, const std::vector<T>& data)
    : tensorImpl_(createTensorAdapter(shape, data.data(), dtype_trait<T>::value)) {}

    template <typename T>
    T Tensor::at(const std::vector<size_t>& indices) const {
        if (!isSameType<T>()) {
            throw std::invalid_argument("Tensor::at: requested type does not match tensor type");
        }
        return static_cast<T>(tensorImpl_->getValueAsDouble(computeFlatIndex(shape(), indices)));
    }

    template <typename T>
    T Tensor::at(size_t index) const {
        if (!isSameType<T>()) {
            throw std::invalid_argument("Tensor::at: requested type does not match tensor type");
        }
        return static_cast<T>(tensorImpl_->getValueAsDouble(index));
    }

    template <typename T>
    bool Tensor::isSameType() const {
        return type() == dtype_trait<T>::value;
    }

    template <typename T>
    const T& Tensor::getImpl() const  {
        const T* impl = dynamic_cast<const T*>(tensorImpl_.get());
        if (!impl) {
            throw std::runtime_error("Invalid tensor implementation type");
        }
        return *impl;
    }

}; // namespace sdnn
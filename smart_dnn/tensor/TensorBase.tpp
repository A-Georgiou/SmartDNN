

namespace sdnn {

    template <typename T>
    Tensor::Tensor(const Shape& shape, const std::vector<T>& data, dtype type)
    : tensorImpl_(createTensorAdapter(shape, data.data(), type)) {}

    template <typename T>
    Tensor::Tensor(const Shape& shape, const std::vector<T>& data)
    : tensorImpl_(createTensorAdapter(shape, data)) {}

    template <typename T>
    Tensor::Tensor(const Shape& shape, const T data, dtype type)
    : tensorImpl_(createTensorAdapter(shape, data, type)) {}

    template <typename T>
    Tensor::Tensor(const Shape& shape, const T data)
    : tensorImpl_(createTensorAdapter(shape, data, dtype_trait<T>::value)) {}

    template <typename T>
    Tensor::Tensor(const Shape& shape, std::initializer_list<T> values) 
    : tensorImpl_(createTensorAdapter(shape, values, dtype_trait<T>::value)) {}

    template <typename T>
    void Tensor::set(size_t index, T&& data) {
        T temp = std::forward<T>(data);
        tensorImpl_->set(index, {static_cast<void*>(&temp), dtype_trait<T>::value});
    }

    template <typename T>
    void Tensor::set(const std::vector<size_t>& indices, T&& data){
        T temp = std::forward<T>(data);
        tensorImpl_->set(indices, {static_cast<void*>(&temp), dtype_trait<T>::value});
    }

    template <typename T>
    T Tensor::at(const std::vector<size_t>& indices) const {
        T out;
        getValueAsType(computeFlatIndex(shape(), indices), &out);
        return out;
    }

    template <typename T>
    T Tensor::at(size_t index) const {
        T out;
        getValueAsType(index, &out);
        return out;
    }

    template <typename T>
    void Tensor::getValueAsType(size_t index, T* out) const {
        tensorImpl_->getValueAsType(index, {out, dtype_trait<T>::value});
    }

    template <typename T>
    Tensor fill(const Shape& shape, const T& fillValue, dtype type) {
        return defaultTensorBackend().fill(shape, {fillValue, dtype_trait<T>::value}, type);
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
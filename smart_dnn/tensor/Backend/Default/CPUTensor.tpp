namespace sdnn {
    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const std::vector<T>& data)
        : shape_(shape), type_(dtype_trait<T>::value) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        allocateMemory();
        std::memcpy(data_.get(), data.data(), shape.size() * sizeof(T));
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const std::vector<T>& data, dtype type)
        : shape_(shape), type_(type) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        allocateMemory();
        for (size_t i = 0; i < data.size(); ++i) {
            convert_dtype(this->data_.get() + i * dtype_size(type), &data[i], type, dtype_trait<T>::value);
        }
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const T* data, dtype type)
        : shape_(shape), type_(type) {
        for (size_t i = 0; i < shape.size(); ++i) {
            convert_dtype(this->data_.get() + i * dtype_size(type), &data[i], type, dtype_trait<T>::value);
        }
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, T data, dtype type)
        : shape_(shape), type_(type) {
        allocateMemory();
        fill({&data, type_});
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, T data)
        : shape_(shape), type_(dtype_trait<T>::value){
        allocateMemory();
        fill({&data, type_});
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const std::initializer_list<T> values, dtype type)
        : shape_(shape), type_(type) {
        if (shape.size() != values.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        allocateMemory();
        for (size_t i = 0; i < values.size(); ++i) {
            auto it = values.begin();
            std::advance(it, i);
            convert_dtype(this->data_.get() + i * dtype_size(type), &(*it), type, dtype_trait<T>::value);
        }
    }

    template <typename AccessOp>
    void CPUTensor::accessData(AccessOp op) const {
        if (index_) {
            #pragma omp parallel for
            for (size_t i = 0; i < shape_.size(); ++i) {
                std::vector<size_t> indices = unflattenIndex(i, shape_);
                size_t flatIndex = index_->flattenIndex(indices);
                op(flatIndex);
            }
        } else {
            #pragma omp parallel for
            for (size_t i = 0; i < shape_.size(); ++i) {
                op(i);
            }
        }
    }

    template<typename TypedOp>
    void CPUTensor::applyTypedOperation(TypedOp op) {
        applyTypedOperationHelper(type_, [this, &op](auto dummy) {
            using T = decltype(dummy);
            op(reinterpret_cast<T*>(data_.get()));
        });
    }

    template<typename TypedOp>
    void CPUTensor::applyTypedOperation(TypedOp op) const {
        applyTypedOperationHelper(type_, [this, &op](auto dummy) {
            using T = decltype(dummy);
            op(reinterpret_cast<const T*>(data_.get()));
        });
    }

    template<typename CompOp>
    bool CPUTensor::elementWiseComparison(const Tensor& other, CompOp op) const {
        if (shape_ != other.shape() || type_ != other.type()) {
            return false;
        }
        const CPUTensor& otherCPU = dynamic_cast<const CPUTensor&>(*other.tensorImpl_);
        
        bool result = true;
        for (size_t i = 0; i < shape_.size(); ++i) {
            applyTypedOperationHelper(type_, [&](auto dummy) {
                using T = decltype(dummy);
                const T* a = reinterpret_cast<const T*>(data_.get()) + i;
                const T* b = reinterpret_cast<const T*>(otherCPU.data_.get()) + i;
                if (!op(*a, *b)) {
                    result = false;
                    return; 
                }
            });
        }
        return result;
    }


    template<typename Op>
    void CPUTensor::elementWiseOperation(const Tensor& other, Op op) {
        if (shape_ != other.shape() || type_ != other.type()) {
            throw std::invalid_argument("Tensor shapes or types do not match");
        }

        if (other.shape().size() != shape_.size()) {
            throw std::invalid_argument("Tensor shapes or types do not match");
        }

        // Ensure we have the correct other tensor to work with
        const CPUTensor& otherCPU = dynamic_cast<const CPUTensor&>(*other.tensorImpl_);

        // Access each element in this tensor (a) and the corresponding element in other (b)
        accessData([&](size_t i) {
            applyTypedOperationHelper(type_, [&](auto dummy) {
                using T = decltype(dummy);
                T* a = reinterpret_cast<T*>(data_.get()) + i;  // Correctly accessing element 'a'
    
                size_t bIndex = 0;
                if (otherCPU.index_) {
                    std::vector<size_t> indices = unflattenIndex(i, shape_);
                    bIndex = otherCPU.index_->flattenIndex(indices);
                }

                const T* b = reinterpret_cast<const T*>(otherCPU.data_.get()) + bIndex;  // Correctly accessing element 'b'
                op(*a, *b);  // Perform the element-wise operation
            });
        });
    }

    template<typename Op>
    void CPUTensor::scalarOperation(double scalar, Op op) {
        accessData([&](size_t i) {
            applyTypedOperationHelper(type_, [&](auto dummy) {
                using T = decltype(dummy);
                T* a = reinterpret_cast<T*>(data_.get()) + i;
                op(*a, static_cast<T>(scalar));
            });
        });
    }

    template<typename TargetType, typename SourceType>
    void CPUTensor::writeElement(void* buffer, size_t index, SourceType value) {
        TargetType* typedBuffer = static_cast<TargetType*>(buffer);
        typedBuffer[index] = static_cast<TargetType>(value);
    }

}; // namespace sdnn
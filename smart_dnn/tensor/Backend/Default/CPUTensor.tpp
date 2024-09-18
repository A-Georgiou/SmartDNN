namespace sdnn {
    template <typename AccessOp>
    void CPUTensor::accessData(AccessOp op) const {
        if (index_) {
            for (size_t i = 0; i < shape_.size(); ++i) {
                std::vector<size_t> indices = unflattenIndex(i, shape_);
                size_t flatIndex = index_->flattenIndex(indices);
                op(flatIndex);
            }
        } else {
            for (size_t i = 0; i < shape_.size(); ++i) {
                op(i);
            }
        }
    }

    template<typename TypedOp>
    void CPUTensor::applyTypedOperation(TypedOp op) {
        applyTypedOperationHelper(type_, [this, &op](auto dummy) {
            using T = decltype(dummy);
            op(reinterpret_cast<T*>((*data_).data()));
        });
    }

    template<typename TypedOp>
    void CPUTensor::applyTypedOperation(TypedOp op) const {
        applyTypedOperationHelper(type_, [this, &op](auto dummy) {
            using T = decltype(dummy);
            op(reinterpret_cast<const T*>((*data_).data()));
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
                const T* a = reinterpret_cast<const T*>(data_->data()) + i;
                const T* b = reinterpret_cast<const T*>(otherCPU.data_->data()) + i;
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
        const CPUTensor& otherCPU = dynamic_cast<const CPUTensor&>(*other.tensorImpl_);
        accessData([&](size_t i) {
            applyTypedOperationHelper(type_, [&](auto dummy) {
                using T = decltype(dummy);
                T* a = reinterpret_cast<T*>(data_->data()) + i;
                const T* b = reinterpret_cast<const T*>(otherCPU.data_->data()) + i;
                op(*a, *b);
            });
        });
    }

    template<typename Op>
    void CPUTensor::scalarOperation(double scalar, Op op) {
        accessData([&](size_t i) {
            applyTypedOperationHelper(type_, [&](auto dummy) {
                using T = decltype(dummy);
                T* a = reinterpret_cast<T*>(data_->data()) + i;
                op(*a, static_cast<T>(scalar));
            });
        });
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const std::vector<T>& data)
        : shape_(shape), type_(dtype_trait<T>::value) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        allocateMemory();
        std::memcpy((*data_).data(), data.data(), shape.size() * sizeof(T));
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const std::vector<T>& data, dtype type)
        : shape_(shape), type_(type) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        allocateMemory();
        for (size_t i = 0; i < data.size(); ++i) {
            convert_dtype(this->data_->data() + i * dtype_size(type), &data[i], type, dtype_trait<T>::value);
        }
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const T* data, dtype type)
        : shape_(shape), type_(type) {
        for (size_t i = 0; i < shape.size(); ++i) {
            convert_dtype(this->data_->data() + i * dtype_size(type), &data[i], type, dtype_trait<T>::value);
        }
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const T& data, dtype type)
        : shape_(shape), type_(type) {
        allocateMemory();
        fill({data, type_});
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const T& data)
        : shape_(shape), type_(dtype_trait<T>::value){
        allocateMemory();
        fill({data, type_});
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const std::initializer_list<T> values, dtype type)
        : shape_(shape), type_(type) {
        if (shape.size() != values.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        allocateMemory();
        for (size_t i = 0; i < values.size(); ++i) {
            convert_dtype(this->data_->data() + i * dtype_size(type), &values[i], type, dtype_trait<T>::value);
        }
    }

    template<typename TargetType, typename SourceType>
    void CPUTensor::writeElement(void* buffer, size_t index, SourceType value) {
        TargetType* typedBuffer = static_cast<TargetType*>(buffer);
        typedBuffer[index] = static_cast<TargetType>(value);
    }

}; // namespace sdnn
namespace sdnn {
    template<typename TypedOp>
    void applyTypedOperationHelper(dtype type, TypedOp op) {
        switch (type) {
            case dtype::f32: op(float{}); break;
            case dtype::f64: op(double{}); break;
            case dtype::b8: op(bool{}); break;
            case dtype::s16: op(int16_t{}); break;
            case dtype::s32: op(int32_t{}); break;
            case dtype::s64: op(int64_t{}); break;
            case dtype::u8: op(uint8_t{}); break;
            case dtype::u16: op(uint16_t{}); break;
            case dtype::u32: op(uint32_t{}); break;
            case dtype::u64: op(uint64_t{}); break;
            default: throw std::runtime_error("Unsupported dtype for operation");
        }
    }

    template <typename AccessOp>
    void CPUTensor::accessData(AccessOp op) const {
        if (index_) {
            // Use TensorIndex for access
            for (size_t i = 0; i < shape_.size(); ++i) {
                std::vector<size_t> indices = unflattenIndex(i, shape_);
                size_t flatIndex = index_->flattenIndex(indices);
                op(flatIndex);
            }
        } else {
            // Direct access without TensorIndex
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
        accessData([&](size_t i) {
            applyTypedOperationHelper(type_, [&](auto dummy) {
                using T = decltype(dummy);
                const T* a = reinterpret_cast<const T*>(data_->data()) + i;
                const T* b = reinterpret_cast<const T*>(otherCPU.data_->data()) + i;
                if (!op(*a, *b)) {
                    result = false;
                    return;  // Early exit if comparison fails
                }
            });
            return result;  // Continue iteration if result is still true
        });
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

    template<typename T>
    void CPUTensor::fill(T value) {
        applyTypedOperationHelper(type_, [this, value](auto dummy) {
            using TargetType = decltype(dummy);
            TargetType* data = reinterpret_cast<TargetType*>(this->data_->data());
            TargetType typedValue;
            
            if constexpr (std::is_pointer_v<T>) {
                typedValue = static_cast<TargetType>(*value);
            } else {
                typedValue = static_cast<TargetType>(value);
            }
            
            std::fill_n(data, shape_.size(), typedValue);
        });
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const std::vector<T>& data)
        : shape_(shape), type_(dtype_trait<T>::value) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        allocateMemory(shape.size() * sizeof(T));
        std::memcpy((*data_).data(), data.data(), shape.size() * sizeof(T));
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const std::vector<T>& data, dtype type)
        : shape_(shape), type_(type), data_(std::make_shared<std::vector<char>>()) {
        if (shape.size() != data.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        allocateMemory(shape.size() * dtype_size(type));
        applyTypedOperationHelper(type, [this, &data](auto dummy) {
            using TargetType = decltype(dummy);
            for (size_t i = 0; i < data.size(); ++i) {
                writeElement<TargetType>(this->data_->data(), i, data[i]);
            }
        });
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const T* data, dtype type)
        : shape_(shape), type_(type), data_(std::make_shared<std::vector<char>>()) {
        initializeData(data, shape.size());
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const T data, dtype type)
        : shape_(shape), type_(type), data_(std::make_shared<std::vector<char>>()) {
        allocateMemory(shape.size() * dtype_size(type));
        fill(data);
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const T data)
        : shape_(shape), type_(dtype_trait<T>::value), data_(std::make_shared<std::vector<char>>()) {
        allocateMemory(shape.size() * dtype_size(type_));
        fill(data);
    }

    template <typename T>
    CPUTensor::CPUTensor(const Shape& shape, const std::initializer_list<T> values, dtype type)
        : shape_(shape), type_(type), data_(std::make_shared<std::vector<char>>()) {
        if (shape.size() != values.size()) {
            throw std::invalid_argument("Data size does not match shape");
        }
        allocateMemory(shape.size() * dtype_size(type_));
        applyTypedOperationHelper(type_, [this, values](auto dummy) {
            using TargetType = decltype(dummy);
            TargetType* data = reinterpret_cast<TargetType*>(this->data_->data());
            size_t i = 0;
            for (const auto& value : values) {
                data[i++] = static_cast<TargetType>(value);
            }
        });
    }

    template<typename T>
    void CPUTensor::setValueAtIndex(size_t index, T&& value) {
        if (index >= shape_.size()) {
            throw std::out_of_range("Index out of range");
        }

        applyTypedOperationHelper(type_, [this, index, value](auto dummy) {
            using TargetType = decltype(dummy);
            writeElement<TargetType, T>(data_->data(), index, value);
        });
    }

    template<typename TargetType, typename SourceType>
    void CPUTensor::writeElement(void* buffer, size_t index, SourceType value) {
        TargetType* typedBuffer = static_cast<TargetType*>(buffer);
        typedBuffer[index] = static_cast<TargetType>(value);
    }

    template <typename T>
    void CPUTensor::initializeData(const T* data, size_t total_elements) {
        size_t type_size = dtype_size(type_);
        size_t total_size = total_elements * type_size;
        
        allocateMemory(total_size);
        
        if (data != nullptr) {
            applyTypedOperationHelper(type_, [this, data, total_elements](auto dummy) {
                using TargetType = decltype(dummy);
                TargetType* dest_data = reinterpret_cast<TargetType*>(this->data_->data());
                for (size_t i = 0; i < total_elements; ++i) {
                    dest_data[i] = static_cast<TargetType>(data[i]);
                }
            });
        } else {
            std::memset(data_->data(), 0, total_size);
        }
    }
}; // namespace sdnn
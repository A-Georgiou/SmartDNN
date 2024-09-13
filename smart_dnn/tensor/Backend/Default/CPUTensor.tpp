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
            return false;  // Tensors with different shapes or types are not equal
        }
        const CPUTensor& otherCPU = dynamic_cast<const CPUTensor&>(*other.tensorImpl_);
        
        bool result = true;
        applyTypedOperation([&](auto* type_ptr) {
            using T = std::remove_pointer_t<decltype(type_ptr)>;
            const auto* a = this->typedData<T>();
            const auto* b = otherCPU.typedData<T>();
            for (size_t i = 0; i < shape_.size(); ++i) {
                if (!op(a[i], b[i])) {
                    result = false;
                    return;  // Early exit if comparison fails
                }
            }
        });
        return result;
    }


    template<typename Op>
    void CPUTensor::elementWiseOperation(const Tensor& other, Op op) {
        if (shape_ != other.shape() || type_ != other.type()) {
            throw std::invalid_argument("Tensor shapes or types do not match");
        }
        const CPUTensor& otherCPU = dynamic_cast<const CPUTensor&>(*other.tensorImpl_);
        
        applyTypedOperation([&](auto* type_ptr) {
            using T = std::remove_pointer_t<decltype(type_ptr)>;
            auto* a = typedData<T>();
            const auto* b = otherCPU.typedData<T>();
            for (size_t i = 0; i < shape_.size(); ++i) {
                op(a[i], b[i]);
            }
        });
    }

    template<typename Op>
    void CPUTensor::scalarOperation(double scalar, Op op) {
        applyTypedOperation([&](auto* type_ptr) {
            using T = std::remove_pointer_t<decltype(type_ptr)>;
            auto* a = typedData<T>();
            for (size_t i = 0; i < shape_.size(); ++i) {
                op(a[i], static_cast<T>(scalar));
            }
        });
    }

    template<typename T>
    void CPUTensor::fill(T value) {
        std::fill(typedData<T>(), typedData<T>() + shape_.size(), value);
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

    template<typename T>
    CPUTensor::CPUTensor(const Shape& shape, const T* data, dtype type)
        : shape_(shape), type_(type) {
        size_t total_elements = shape.size();
        allocateMemory(total_elements * sizeof(T));
        std::memcpy((*data_).data(), data, total_elements * sizeof(T));
    }

    template<typename TargetType>
    void writeElement(void* buffer, size_t index, double value) {
        static_cast<TargetType*>(buffer)[index] = static_cast<TargetType>(value);
    }
}; // namespace sdnn
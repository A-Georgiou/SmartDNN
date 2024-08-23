#ifndef TENSOR_OPERATIONS_HPP
#define TENSOR_OPERATIONS_HPP

template <typename T>
class TensorOperations {
public:
    TensorOperations() = delete;

    static void add(TensorData<T>& lhs, const TensorData<T>& rhs);
    static void subtract(TensorData<T>& lhs, const TensorData<T>& rhs);
    static void multiply(TensorData<T>& lhs, const TensorData<T>& rhs);
    static void divide(TensorData<T>& lhs, const TensorData<T>& rhs);

    static T sum(const TensorData<T>& tensor);
    static TensorData<T> sqrt(const TensorData<T>& tensor);
    static TensorData<T> apply(const TensorData<T>& tensor, std::function<T(T)> op);

    // Other operations...
};

#endif // TENSOR_OPERATIONS_HPP
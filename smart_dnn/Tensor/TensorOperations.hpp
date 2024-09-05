#ifndef TENSOR_OPERATIONS_HPP
#define TENSOR_OPERATIONS_HPP

#include "TensorData.hpp"
#include "BroadcastView.hpp"
#include "../RandomEngine.hpp"
#include <functional>

namespace smart_dnn {

#if __cplusplus >= 202002L
// use concept (C++20)
template<typename T, typename DeviceType>
concept TensorLike = std::is_same_v<T, TensorData<typename T::value_type, DeviceType>> ||
                     std::is_same_v<T, BroadcastView<typename T::value_type, DeviceType>>;
#else
// use type trait (C++17)
template<typename T, typename DeviceType>
struct is_tensor_like : std::false_type {};

template<typename T, typename DeviceType>
struct is_tensor_like<TensorData<T, DeviceType>, DeviceType> : std::true_type {};

template<typename T, typename DeviceType>
struct is_tensor_like<BroadcastView<T, DeviceType>, DeviceType> : std::true_type {};
#endif

template <typename T, typename DeviceType>
class TensorOperations {
public:
    TensorOperations() = delete; // Prevent instantiation

    // Non-in-place operations
    static TensorData<T, DeviceType> add(const TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);
    static TensorData<T, DeviceType> subtract(const TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);
    static TensorData<T, DeviceType> multiply(const TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);
    static TensorData<T, DeviceType> divide(const TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);

    // In-place operations
    static TensorData<T, DeviceType>& addInPlace(TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);
    static TensorData<T, DeviceType>& subtractInPlace(TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);
    static TensorData<T, DeviceType>& multiplyInPlace(TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);
    static TensorData<T, DeviceType>& divideInPlace(TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);

    // Scalar operations
    static TensorData<T, DeviceType> addScalar(const TensorData<T, DeviceType>& tensor, T scalar);
    static TensorData<T, DeviceType> subtractScalar(const TensorData<T, DeviceType>& tensor, T scalar);
    static TensorData<T, DeviceType> multiplyScalar(const TensorData<T, DeviceType>& tensor, T scalar);
    static TensorData<T, DeviceType> divideScalar(const TensorData<T, DeviceType>& tensor, T scalar);

    // Special scalar divide operation
    static TensorData<T, DeviceType> inverseDivideScalar(const TensorData<T, DeviceType>& tensor, T scalar);

    // In-place scalar operations
    static TensorData<T, DeviceType>& addScalarInPlace(TensorData<T, DeviceType>& tensor, T scalar);
    static TensorData<T, DeviceType>& subtractScalarInPlace(TensorData<T, DeviceType>& tensor, T scalar);
    static TensorData<T, DeviceType>& multiplyScalarInPlace(TensorData<T, DeviceType>& tensor, T scalar);
    static TensorData<T, DeviceType>& divideScalarInPlace(TensorData<T, DeviceType>& tensor, T scalar);

    // Other operations
    static TensorData<T, DeviceType> apply(const TensorData<T, CPUDevice>& tensor, std::function<T(T)> func);
    static TensorData<T, DeviceType>& applyInPlace(TensorData<T, DeviceType>& tensor, std::function<T(T)> func);
    static TensorData<T, CPUDevice> sum(const TensorData<T, CPUDevice>& tensor);
    static TensorData<T, CPUDevice> sum(const TensorData<T, CPUDevice>& tensor, int axis);
    static TensorData<T, DeviceType> sqrt(const TensorData<T, DeviceType>& tensor);
    static TensorData<T, DeviceType>& sqrtInPlace(TensorData<T, DeviceType>& tensor);

    // Generative operations
    static TensorData<T, DeviceType> createFill(const Shape& shape, T value);
    static TensorData<T, DeviceType> createRandom(const Shape& shape, T min, T max);
    static TensorData<T, DeviceType> createIdentity(int size);
};

// Specialization for CPUDevice
template <typename T>
class TensorOperations<T, CPUDevice> {
public:
    TensorOperations() = delete; // Prevent instantiation

    static TensorData<T, CPUDevice> add(const TensorData<T, CPUDevice>& lhs, const TensorData<T, CPUDevice>& rhs){
        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        TensorData<T, CPUDevice> result(result_shape);

        BroadcastView<T, CPUDevice> lhs_broadcast(lhs, result_shape);
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);   
    
        addImpl(lhs_broadcast, rhs_broadcast, result);
        return result;
    }

    static TensorData<T, CPUDevice> subtract(const TensorData<T, CPUDevice>& lhs, const TensorData<T, CPUDevice>& rhs){
        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        TensorData<T, CPUDevice> result(result_shape);

        BroadcastView<T, CPUDevice> lhs_broadcast(lhs, result_shape);
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        minusImpl(lhs_broadcast, rhs_broadcast, result);
        return result;
    }

    static TensorData<T, CPUDevice> multiply(const TensorData<T, CPUDevice>& lhs, const TensorData<T, CPUDevice>& rhs){
        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        TensorData<T, CPUDevice> result(result_shape);

        BroadcastView<T, CPUDevice> lhs_broadcast(lhs, result_shape);
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        multiplyImpl(lhs_broadcast, rhs_broadcast, result);
        return result;
    }

    static TensorData<T, CPUDevice> divide(const TensorData<T, CPUDevice>& lhs, const TensorData<T, CPUDevice>& rhs){
        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        TensorData<T, CPUDevice> result(result_shape);

        BroadcastView<T, CPUDevice> lhs_broadcast(lhs, result_shape);
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        divideImpl(lhs_broadcast, rhs_broadcast, result);
        return result;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice>& addInPlace(LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");

        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        addImpl(lhs, rhs_broadcast, lhs);
        return lhs;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice>& subtractInPlace(LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");

        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        minusImpl(lhs, rhs_broadcast, lhs);
        return lhs;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice>& multiplyInPlace(LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");
        
        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        multiplyImpl(lhs, rhs_broadcast, lhs);
        return lhs;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice>& divideInPlace(LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");
        
        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        divideImpl(lhs, rhs_broadcast, lhs);
        return lhs;
    }

    static TensorData<T, CPUDevice> addScalar(const TensorData<T, CPUDevice>& tensor, T scalar){
        TensorData<T, CPUDevice> result(tensor.shape());
        addScalarImpl(tensor, scalar, result);
        return result;
    }

    static TensorData<T, CPUDevice> subtractScalar(const TensorData<T, CPUDevice>& tensor, T scalar){
        TensorData<T, CPUDevice> result(tensor.shape());
        subtractScalarImpl(tensor, scalar, result);
        return result;
    }

    static TensorData<T, CPUDevice> multiplyScalar(const TensorData<T, CPUDevice>& tensor, T scalar){
        TensorData<T, CPUDevice> result(tensor.shape());
        multiplyScalarImpl(tensor, scalar, result);
        return result;
    }

    static TensorData<T, CPUDevice> divideScalar(const TensorData<T, CPUDevice>& tensor, T scalar){
        TensorData<T, CPUDevice> result(tensor.shape());
        divideScalarImpl(tensor, scalar, result);
        return result;
    }

    static TensorData<T, CPUDevice> inverseDivideScalar(const TensorData<T, CPUDevice>& tensor, T scalar){
        TensorData<T, CPUDevice> result(tensor.shape(), scalar);
        divideImpl(result, tensor, result);
        return result;
    }

    static TensorData<T, CPUDevice>& addScalarInPlace(TensorData<T, CPUDevice>& tensor, T scalar){
        addScalarImpl(tensor, scalar, tensor);
        return tensor;
    }

    static TensorData<T, CPUDevice>& subtractScalarInPlace(TensorData<T, CPUDevice>& tensor, T scalar){
        subtractScalarImpl(tensor, scalar, tensor);
        return tensor;
    }

    static TensorData<T, CPUDevice>& multiplyScalarInPlace(TensorData<T, CPUDevice>& tensor, T scalar){
        multiplyScalarImpl(tensor, scalar, tensor);
        return tensor;
    }

    static TensorData<T, CPUDevice>& divideScalarInPlace(TensorData<T, CPUDevice>& tensor, T scalar){
        divideScalarImpl(tensor, scalar, tensor);
        return tensor;
    }

    static TensorData<T, CPUDevice> apply(const TensorData<T, CPUDevice>& tensor, std::function<T(T)> func){
        TensorData<T, CPUDevice> result(tensor.shape());
        applyImpl(tensor, result, func);
        return result;
    }

    static TensorData<T, CPUDevice>& applyInPlace(TensorData<T, CPUDevice>& tensor, std::function<T(T)> func){
        applyImpl(tensor, tensor, func);
        return tensor;
    }

    static TensorData<T, CPUDevice> sum(const TensorData<T, CPUDevice>& tensor){
        T result = T(0);
        for (const auto& value : tensor) {
            result += value;
        }
        return TensorData<T, CPUDevice>(Shape({1}), result);
    }

    static TensorData<T, CPUDevice> sum(const TensorData<T, CPUDevice>& tensor, size_t axis) {
        if (axis < 0 || axis >= tensor.shape().rank()) {
            throw std::runtime_error("Invalid axis for sum operation, axis: " + std::to_string(axis) + ", tensor rank: " + std::to_string(tensor.shape().rank()));
        }

        std::vector<int> new_shape_dims = tensor.shape().getDimensions();

        new_shape_dims.erase(new_shape_dims.begin() + axis);  // Remove the summed axis dimension
        Shape new_shape(new_shape_dims);

        TensorData<T, CPUDevice> result(new_shape, T(0));

        std::vector<int> indices(tensor.shape().rank(), 0);  
        for (size_t i = 0; i < tensor.size(); ++i) {
            std::vector<int> result_indices;
            for (size_t j = 0; j < tensor.shape().rank(); ++j) {
                if (j != axis) {
                    result_indices.push_back(indices[j]);
                }
            }

            result.at(result_indices) += tensor[i];

            for (int j = tensor.shape().rank() - 1; j >= 0; --j) {
                if (++indices[j] < tensor.shape()[j]) {
                    break; 
                }
                indices[j] = 0; 
            }
        }

        return result;
    }

    static TensorData<T, CPUDevice> sqrt(const TensorData<T, CPUDevice>& tensor){
        TensorData<T, CPUDevice> result(tensor.shape());
        auto result_it = result.begin();
        for (const auto& value : tensor) {
            *result_it = std::sqrt(value);
            ++result_it;
        }
        return result;
    }

    static TensorData<T, CPUDevice>& sqrtInPlace(TensorData<T, CPUDevice>& tensor){
        auto tensor_it = tensor.begin();
        for (auto& value : tensor) {
            value = std::sqrt(value);
            ++tensor_it;
        }
        return tensor;
    }

    static TensorData<T, CPUDevice> createFill(const Shape& shape, T value){
        return TensorData<T, CPUDevice>(shape, value);
    }

    static TensorData<T, CPUDevice> createRandom(const Shape& shape, T min, T max){
        TensorData<T, CPUDevice> result(shape);
        
        #pragma omp parallel for
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = RandomEngine::getHeRandRange(shape.size(), min, max);
        }

        return result;
    }

    static TensorData<T, CPUDevice> createIdentity(int size){
        Shape shape({size, size});
        TensorData<T, CPUDevice> result(shape);
        auto result_it = result.begin();
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                *result_it = (i == j) ? T(1) : T(0);
                ++result_it;
            }
        }
        return result;
    }

private:
    template<typename LHS, typename RHS>
    static constexpr bool are_tensor_like() {
        return is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value;
    }

    template<typename Operation>
    static void applyScalarImpl(const TensorData<T, CPUDevice>& tensor, T scalar, TensorData<T, CPUDevice>& result, Operation op) {
        std::transform(tensor.begin(), tensor.end(), result.begin(), [&](T value) { return op(value, scalar); });
    }

    template<typename LHS, typename RHS>
    static void addImpl(const LHS& lhs, const RHS& rhs, TensorData<T, CPUDevice>& result) {
        static_assert(are_tensor_like<LHS, RHS>(), "Both operands must be either TensorData or BroadcastView");
        if (lhs.shape().size() != rhs.shape().size()) { throw std::runtime_error("Shapes are not broadcastable! Mismatch between shapes: " + lhs.shape().toString() + " + " + rhs.shape().toString()); }
 
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), std::plus<T>());
    }

    template<typename LHS, typename RHS>
    static void minusImpl(const LHS& lhs, const RHS& rhs, TensorData<T, CPUDevice>& result) {
        static_assert(are_tensor_like<LHS, RHS>(), "Both operands must be either TensorData or BroadcastView");
        if (lhs.shape().size() != rhs.shape().size()) { throw std::runtime_error("Shapes are not broadcastable! Mismatch between shapes: " + lhs.shape().toString() + " + " + rhs.shape().toString()); }
    
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), std::minus<T>());
    }

    template<typename LHS, typename RHS>
    static void multiplyImpl(const LHS& lhs, const RHS& rhs, TensorData<T, CPUDevice>& result) {
        static_assert(are_tensor_like<LHS, RHS>(), "Both operands must be either TensorData or BroadcastView");
        if (lhs.shape().size() != rhs.shape().size()) { throw std::runtime_error("Shapes are not broadcastable! Mismatch between shapes: " + lhs.shape().toString() + " + " + rhs.shape().toString()); }

        std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), std::multiplies<T>());
    }

    template<typename LHS, typename RHS>
    static void divideImpl(const LHS& lhs, const RHS& rhs, TensorData<T, CPUDevice>& result) {
        static_assert(are_tensor_like<LHS, RHS>(), "Both operands must be either TensorData or BroadcastView");
        if (lhs.shape().size() != rhs.shape().size()) { throw std::runtime_error("Shapes are not broadcastable! Mismatch between shapes: " + lhs.shape().toString() + " + " + rhs.shape().toString()); }
    
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), std::divides<T>());
    }

    static void addScalarImpl(const TensorData<T, CPUDevice>& tensor, T scalar, TensorData<T, CPUDevice>& result) {
        applyScalarImpl(tensor, scalar, result, std::plus<T>());
    }

    static void subtractScalarImpl(const TensorData<T, CPUDevice>& tensor, T scalar, TensorData<T, CPUDevice>& result) {

        applyScalarImpl(tensor, scalar, result, std::minus<T>());
    }

    static void multiplyScalarImpl(const TensorData<T, CPUDevice>& tensor, T scalar, TensorData<T, CPUDevice>& result) {
        applyScalarImpl(tensor, scalar, result, std::multiplies<T>());
    }

    static void divideScalarImpl(const TensorData<T, CPUDevice>& tensor, T scalar, TensorData<T, CPUDevice>& result) {
        if (scalar == T(0)) {
            throw std::invalid_argument("Division by zero");
        }
        applyScalarImpl(tensor, scalar, result, std::divides<T>());
    }

    static void applyImpl(const TensorData<T, CPUDevice>& tensor, TensorData<T, CPUDevice>& result, std::function<T(T)> func) {
        std::transform(tensor.begin(), tensor.end(), result.begin(), func);
    }
    

};

} // namespace smart_dnn

#endif // TENSOR_OPERATIONS_HPP
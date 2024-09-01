#ifndef TENSOR_OPERATIONS_HPP
#define TENSOR_OPERATIONS_HPP

#include "TensorData.hpp"
#include "BroadcastView.hpp"
#include "../RandomEngine.hpp"

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
    static T sum(const TensorData<T, DeviceType>& tensor);
    static TensorData<T, DeviceType> sqrt(const TensorData<T, DeviceType>& tensor);
    static TensorData<T, DeviceType>& sqrtInPlace(TensorData<T, DeviceType>& tensor);

    // Generative operations
    static TensorData<T, DeviceType> createFill(const Shape& shape, T value);
    static TensorData<T, DeviceType> createRandom(const Shape& shape, T min, T max);
};

// Specialization for CPUDevice
template <typename T>
class TensorOperations<T, CPUDevice> {
public:
    TensorOperations() = delete; // Prevent instantiation

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice> add(const LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");

        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        TensorData<T, CPUDevice> result(result_shape);

        BroadcastView<T, CPUDevice> lhs_broadcast(lhs, result_shape);
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        addImpl(lhs, rhs, result);
        return result;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice> subtract(const LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");

        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        TensorData<T, CPUDevice> result(result_shape);

        BroadcastView<T, CPUDevice> lhs_broadcast(lhs, result_shape);
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        subtractImpl(lhs, rhs, result);
        return result;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice> multiply(const LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");

        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        TensorData<T, CPUDevice> result(result_shape);

        BroadcastView<T, CPUDevice> lhs_broadcast(lhs, result_shape);
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        multiplyImpl(lhs_broadcast, rhs_broadcast, result);
        return result;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice> divide(const LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");

        Shape result_shape = ShapeOperations::broadcastShapes(lhs.shape(), rhs.shape());
        TensorData<T, CPUDevice> result(result_shape);

        BroadcastView<T, CPUDevice> lhs_broadcast(lhs, result_shape);
        BroadcastView<T, CPUDevice> rhs_broadcast(rhs, result_shape);

        divideImpl(lhs, rhs, result);
        return result;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice>& addInPlace(LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");

        addImpl(lhs, rhs, lhs);
        return lhs;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice>& subtractInPlace(LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");

        subtractImpl(lhs, rhs, lhs);
        return lhs;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice>& multiplyInPlace(LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");

        multiplyImpl(lhs, rhs, lhs);
        return lhs;
    }

    template<typename LHS, typename RHS>
    static TensorData<T, CPUDevice>& divideInPlace(LHS& lhs, const RHS& rhs){
        static_assert(is_tensor_like<LHS, CPUDevice>::value && is_tensor_like<RHS, CPUDevice>::value,
                      "Both operands must be either TensorData or BroadcastView");

        divideImpl(lhs, rhs, lhs);
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

    static T sum(const TensorData<T, CPUDevice>& tensor){
        T result = 0;
        for (const auto& value : tensor) {
            result += value;
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
        return {shape, value};
    }

    static TensorData<T, CPUDevice> createRandom(const Shape& shape, T min, T max){
        TensorData<T, CPUDevice> result(shape);
        
        auto tensor_it = result.begin();
        while (tensor_it != result.end()) {
            *tensor_it = RandomEngine::getRandRange(min, max);
            ++tensor_it;
        }
    }

private:
    static void addImpl(const TensorData<T, CPUDevice>& lhs,
                        const TensorData<T, CPUDevice>& rhs, TensorData<T,
                        CPUDevice>& result) {
        auto lhs_it = lhs.begin();
        auto rhs_it = rhs.begin();
        auto result_it = result.begin();

        while (result_it != result.end()) {
            *result_it = *lhs_it + *rhs_it;
            ++lhs_it;
            ++rhs_it;
            ++result_it;
        }
    }

    static void minusImpl(const TensorData<T, CPUDevice>& lhs,
                          const TensorData<T, CPUDevice>& rhs, TensorData<T,
                          CPUDevice>& result) {
        auto lhs_it = lhs.begin();
        auto rhs_it = rhs.begin();
        auto result_it = result.begin();

        while (result_it != result.end()) {
            *result_it = *lhs_it - *rhs_it;
            ++lhs_it;
            ++rhs_it;
            ++result_it;
        }
    }

    template<typename LHS, typename RHS>
    static void multiplyImpl(const LHS& lhs,
                             const RHS& rhs,
                             TensorData<T, CPUDevice>& result) {
        auto lhs_it = lhs.begin();
        auto rhs_it = rhs.begin();
        auto result_it = result.begin();

        while (result_it != result.end()) {
            *result_it = *lhs_it * *rhs_it;
            ++lhs_it;
            ++rhs_it;
            ++result_it;
        }
    }

    static void divideImpl(const TensorData<T, CPUDevice>& lhs,
                           const TensorData<T, CPUDevice>& rhs, TensorData<T,
                           CPUDevice>& result) {
        auto lhs_it = lhs.begin();
        auto rhs_it = rhs.begin();
        auto result_it = result.begin();

        while (result_it != result.end()) {
            *result_it = *lhs_it / *rhs_it;
            ++lhs_it;
            ++rhs_it;
            ++result_it;
        }
    }

    static void addScalarImpl(const TensorData<T, CPUDevice>& tensor, T scalar,
                              TensorData<T, CPUDevice>& result) {
        auto tensor_it = tensor.begin();
        auto result_it = result.begin();

        while (result_it != result.end()) {
            *result_it = *tensor_it + scalar;
            ++tensor_it;
            ++result_it;
        }
    }

    static void subtractScalarImpl(const TensorData<T, CPUDevice>& tensor, T scalar,
                                   TensorData<T, CPUDevice>& result) {
        auto tensor_it = tensor.begin();
        auto result_it = result.begin();

        while (result_it != result.end()) {
            *result_it = *tensor_it - scalar;
            ++tensor_it;
            ++result_it;
        }
    }

    static void multiplyScalarImpl(const TensorData<T, CPUDevice>& tensor, T scalar,
                                   TensorData<T, CPUDevice>& result) {
        auto tensor_it = tensor.begin();
        auto result_it = result.begin();

        while (result_it != result.end()) {
            *result_it = *tensor_it * scalar;
            ++tensor_it;
            ++result_it;
        }
    }

    static void divideScalarImpl(const TensorData<T, CPUDevice>& tensor, T scalar,
                                 TensorData<T, CPUDevice>& result) {
        auto tensor_it = tensor.begin();
        auto result_it = result.begin();

        while (result_it != result.end()) {
            *result_it = *tensor_it / scalar;
            ++tensor_it;
            ++result_it;
        }
    }
    

};

} // namespace smart_dnn

#endif // TENSOR_OPERATIONS_HPP
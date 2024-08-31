#ifndef TENSOR_OPERATIONS_HPP
#define TENSOR_OPERATIONS_HPP

#include "TensorData.hpp"

// General template definition for TensorOperations
template <typename T, typename DeviceType>
class TensorOperations {
public:
    TensorOperations() = delete; // Prevent instantiation

    // These methods should be defined appropriately depending on DeviceType
    static void add(TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);
    static void subtract(TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);
    static void multiply(TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);
    static void divide(TensorData<T, DeviceType>& lhs, const TensorData<T, DeviceType>& rhs);

    static void addScalar(TensorData<T, DeviceType>& tensor, T scalar);
    static void subtractScalar(TensorData<T, DeviceType>& tensor, T scalar);
    static void multiplyScalar(TensorData<T, DeviceType>& tensor, T scalar);
    static void divideScalar(TensorData<T, DeviceType>& tensor, T scalar);

    static T sum(const TensorData<T, DeviceType>& tensor);
    static TensorData<T, DeviceType> sqrt(const TensorData<T, DeviceType>& tensor);
};

// Specialization for CPUDevice
template <typename T>
class TensorOperations<T, CPUDevice> {
public:
    TensorOperations() = delete; // Prevent instantiation

    static void add(TensorData<T, CPUDevice>& lhs, const TensorData<T, CPUDevice>& rhs) {
        
    }

    static void subtract(TensorData<T, CPUDevice>& lhs, const TensorData<T, CPUDevice>& rhs) {
        // Implement CPU-specific subtraction logic
    }

    static void multiply(TensorData<T, CPUDevice>& lhs, const TensorData<T, CPUDevice>& rhs) {
        // Implement CPU-specific multiplication logic
    }

    static void divide(TensorData<T, CPUDevice>& lhs, const TensorData<T, CPUDevice>& rhs) {
        // Implement CPU-specific division logic
    }

    static void addScalar(TensorData<T, CPUDevice>& tensor, T scalar) {
        // Implement CPU-specific scalar addition logic
    }

    static void subtractScalar(TensorData<T, CPUDevice>& tensor, T scalar) {
        // Implement CPU-specific scalar subtraction logic
    }

    static void multiplyScalar(TensorData<T, CPUDevice>& tensor, T scalar) {
        // Implement CPU-specific scalar multiplication logic
    }

    static void divideScalar(TensorData<T, CPUDevice>& tensor, T scalar) {
        // Implement CPU-specific scalar division logic
    }

    static T sum(const TensorData<T, CPUDevice>& tensor) {
        // Implement CPU-specific sum logic
    }

    static TensorData<T, CPUDevice> sqrt(const TensorData<T, CPUDevice>& tensor) {
        // Implement CPU-specific square root logic
    }

private:

};

// You would similarly specialize TensorOperations for other DeviceTypes, like GPUDevice, etc.

#endif // TENSOR_OPERATIONS_HPP
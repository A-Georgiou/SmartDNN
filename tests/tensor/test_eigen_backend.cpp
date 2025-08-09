#include <gtest/gtest.h>
#include "smart_dnn/tensor/Backend/Eigen/EigenTensorBackend.hpp"
#include "smart_dnn/tensor/Backend/Default/CPUTensor.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/shape/Shape.hpp"

namespace sdnn {

TEST(EigenBackendTest, BasicCreation) {
    EigenTensorBackend backend;
    
    // Test backend name
    EXPECT_EQ(backend.backendName(), "Eigen");
}

TEST(EigenBackendTest, ZerosCreation) {
    EigenTensorBackend backend;
    
    Shape shape({2, 3});
    Tensor zeros = backend.zeros(shape, dtype::f32);
    
    EXPECT_EQ(zeros.shape(), shape);
    EXPECT_EQ(zeros.type(), dtype::f32);
}

TEST(EigenBackendTest, OnesCreation) {
    EigenTensorBackend backend;
    
    Shape shape({2, 3});
    Tensor ones = backend.ones(shape, dtype::f32);
    
    EXPECT_EQ(ones.shape(), shape);
    EXPECT_EQ(ones.type(), dtype::f32);
}

TEST(EigenBackendTest, IdentityCreation) {
    EigenTensorBackend backend;
    
    Tensor identity = backend.identity(3, dtype::f32);
    
    EXPECT_EQ(identity.shape(), Shape({3, 3}));
    EXPECT_EQ(identity.type(), dtype::f32);
}

} // namespace sdnn
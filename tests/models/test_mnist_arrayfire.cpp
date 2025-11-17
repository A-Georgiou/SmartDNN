#include <gtest/gtest.h>
#include "smart_dnn/tensor/TensorCreationUtil.hpp"
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/activations/ReLU.hpp"
#include "smart_dnn/activations/Softmax.hpp"
#include "smart_dnn/loss/CategoricalCrossEntropyLoss.hpp"
#include "smart_dnn/layers/FullyConnectedLayer.hpp"
#include "smart_dnn/layers/ActivationLayer.hpp"
#include "smart_dnn/optimizers/AdamOptimizer.hpp"

namespace sdnn {

class MNistArrayFireTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up is called before each test
    }
};

TEST_F(MNistArrayFireTest, SimpleFeedForwardNetwork) {
    // Create a simple 2-layer network for testing
    SmartDNN model;
    
    // Input: 10 features, Output: 5 classes
    model.addLayer(new FullyConnectedLayer(10, 20));
    model.addLayer(new ActivationLayer(new ReLU()));
    model.addLayer(new FullyConnectedLayer(20, 5));
    model.addLayer(new ActivationLayer(new Softmax()));
    
    AdamOptions adamOptions;
    adamOptions.learningRate = 0.001f;
    adamOptions.beta1 = 0.9f;
    adamOptions.beta2 = 0.999f;
    adamOptions.epsilon = 1e-8f;
    
    model.compile(new CategoricalCrossEntropyLoss(), new AdamOptimizer(adamOptions));
    
    // Create dummy input (batch of 2 samples, 10 features each)
    Tensor input = Tensor(Shape({2, 10}), 0.1f);
    
    // Test forward pass
    Tensor output = model.predict(input);
    
    // Output should be (2, 5) - 2 samples, 5 classes
    EXPECT_EQ(output.shape().rank(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 5);
}

TEST_F(MNistArrayFireTest, SingleLayerForward) {
    // Test a single fully connected layer
    FullyConnectedLayer layer(10, 5);
    
    // Initialize with some weights
    Tensor input = Tensor(Shape({2, 10}), 1.0f);  // 2 samples, 10 features
    
    Tensor output = layer.forward(input);
    
    // Output should be (2, 5) - 2 samples, 5 outputs
    EXPECT_EQ(output.shape().rank(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 5);
}

TEST_F(MNistArrayFireTest, ActivationLayerForward) {
    // Test ReLU activation
    ActivationLayer layer(new ReLU());
    
    Tensor input = Tensor(Shape({2, 3}), std::vector<float>{-1.0f, 0.0f, 1.0f, -2.0f, 0.5f, 2.0f});
    
    Tensor output = layer.forward(input);
    
    // Output shape should match input
    EXPECT_EQ(output.shape().rank(), 2);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
}

TEST_F(MNistArrayFireTest, BackendVerification) {
    // Verify we're using the ArrayFire backend
    Tensor a = Tensor(Shape({2, 2}), 1.0f);
    std::string backendName = a.backend().backendName();
    
    EXPECT_TRUE(backendName.find("GPU") != std::string::npos || 
                backendName.find("ArrayFire") != std::string::npos);
}

TEST_F(MNistArrayFireTest, TensorCreationForMNist) {
    // Test creating tensors with MNIST-like dimensions
    // MNIST images are 28x28 = 784 pixels
    Tensor input = Tensor(Shape({1, 784}), 0.5f);
    
    EXPECT_EQ(input.shape().rank(), 2);
    EXPECT_EQ(input.shape()[0], 1);
    EXPECT_EQ(input.shape()[1], 784);
}

TEST_F(MNistArrayFireTest, BatchProcessing) {
    // Test batch processing with multiple samples
    const int batch_size = 4;
    const int input_size = 784;
    
    Tensor batch = Tensor(Shape({batch_size, input_size}), 0.1f);
    
    EXPECT_EQ(batch.shape().rank(), 2);
    EXPECT_EQ(batch.shape()[0], batch_size);
    EXPECT_EQ(batch.shape()[1], input_size);
}

} // namespace sdnn

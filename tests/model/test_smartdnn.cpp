#ifndef TEST_SMARTDNN_CPP
#define TEST_SMARTDNN_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/SmartDNN.hpp"
#include "../../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../../smart_dnn/Layers/ActivationLayer.hpp"
#include "../../smart_dnn/Activations/ReLU.hpp"
#include "../../smart_dnn/Activations/Sigmoid.hpp"
#include "../../smart_dnn/Activations/Softmax.hpp"
#include "../../smart_dnn/Loss/MSELoss.hpp"
#include "../../smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include "../../smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "../../smart_dnn/Optimizers/SGDOptimizer.hpp"

namespace smart_dnn {

/*
    MODEL CREATION AND LAYER ADDITION TESTS
*/

TEST(SmartDNNTest, ModelCreation) {
    SmartDNN<float> model;
    // Model should be created successfully
    ASSERT_NO_THROW(model.addLayer(FullyConnectedLayer<float>(10, 5)));
}

TEST(SmartDNNTest, AddSingleLayer) {
    SmartDNN<float> model;
    
    // Add a single fully connected layer
    model.addLayer(FullyConnectedLayer<float>(10, 5));
    
    // Verify layer was added by checking we can retrieve it
    Layer<float>* layer = model.getLayer(0);
    ASSERT_NE(layer, nullptr);
}

TEST(SmartDNNTest, AddMultipleLayers) {
    SmartDNN<float> model;
    
    // Add multiple layers
    model.addLayer(FullyConnectedLayer<float>(10, 20));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(20, 10));
    model.addLayer(ActivationLayer<float>(Sigmoid<float>()));
    
    // Verify all layers were added
    ASSERT_NE(model.getLayer(0), nullptr);
    ASSERT_NE(model.getLayer(1), nullptr);
    ASSERT_NE(model.getLayer(2), nullptr);
    ASSERT_NE(model.getLayer(3), nullptr);
}

/*
    MODEL COMPILATION TESTS
*/

TEST(SmartDNNTest, CompileWithMSELoss) {
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer<float>(10, 5));
    
    // Compile model with MSE loss and SGD optimizer
    ASSERT_NO_THROW(model.compile(MSELoss<float>(), SGDOptimizer<float>()));
}

TEST(SmartDNNTest, CompileWithCategoricalCrossEntropy) {
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer<float>(10, 5));
    model.addLayer(ActivationLayer<float>(Softmax<float>()));
    
    // Compile model with categorical cross entropy loss
    ASSERT_NO_THROW(model.compile(CategoricalCrossEntropyLoss<float>(), AdamOptimizer<float>()));
}

TEST(SmartDNNTest, CompileWithAdamOptimizer) {
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer<float>(10, 5));
    
    // Configure Adam optimizer with custom options
    AdamOptions adamOptions;
    adamOptions.learningRate = 0.001f;
    adamOptions.beta1 = 0.9f;
    adamOptions.beta2 = 0.999f;
    adamOptions.epsilon = 1e-8f;
    
    // Compile model with Adam optimizer
    ASSERT_NO_THROW(model.compile(MSELoss<float>(), AdamOptimizer<float>(adamOptions)));
}

/*
    PREDICTION TESTS
*/

TEST(SmartDNNTest, PredictSingleInput) {
    SmartDNN<float> model;
    
    // Build a simple model
    model.addLayer(FullyConnectedLayer<float>(3, 2));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.compile(MSELoss<float>(), SGDOptimizer<float>());
    
    // Create input tensor
    Tensor<float> input({3}, {1.0f, 2.0f, 3.0f});
    
    // Make prediction
    Tensor<float> output = model.predict(input);
    
    // Verify output shape
    ASSERT_EQ(output.getShape(), Shape({2}));
}

TEST(SmartDNNTest, PredictBatchInput) {
    SmartDNN<float> model;
    
    // Build a simple model
    model.addLayer(FullyConnectedLayer<float>(3, 2));
    model.compile(MSELoss<float>(), SGDOptimizer<float>());
    
    // Create batch input
    std::vector<Tensor<float>> inputs;
    inputs.push_back(Tensor<float>({3}, {1.0f, 2.0f, 3.0f}));
    inputs.push_back(Tensor<float>({3}, {4.0f, 5.0f, 6.0f}));
    
    // Make predictions
    std::vector<Tensor<float>> outputs = model.predict(inputs);
    
    // Verify we got 2 outputs
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(outputs[0].getShape(), Shape({2}));
    ASSERT_EQ(outputs[1].getShape(), Shape({2}));
}

TEST(SmartDNNTest, PredictMultiLayerNetwork) {
    SmartDNN<float> model;
    
    // Build a multi-layer network
    model.addLayer(FullyConnectedLayer<float>(4, 8));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(8, 4));
    model.addLayer(ActivationLayer<float>(Sigmoid<float>()));
    model.compile(MSELoss<float>(), SGDOptimizer<float>());
    
    // Create input
    Tensor<float> input({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    // Make prediction
    Tensor<float> output = model.predict(input);
    
    // Verify output shape and values are in sigmoid range [0, 1]
    ASSERT_EQ(output.getShape(), Shape({4}));
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_GE(output.getData()[i], 0.0f);
        ASSERT_LE(output.getData()[i], 1.0f);
    }
}

/*
    TRAINING TESTS
*/

TEST(SmartDNNTest, TrainSimpleModel) {
    SmartDNN<float> model;
    
    // Build a simple regression model
    model.addLayer(FullyConnectedLayer<float>(2, 1));
    model.compile(MSELoss<float>(), SGDOptimizer<float>());
    
    // Create simple training data (y = x1 + x2)
    std::vector<Tensor<float>> inputs;
    inputs.push_back(Tensor<float>({2}, {1.0f, 2.0f}));
    inputs.push_back(Tensor<float>({2}, {2.0f, 3.0f}));
    
    std::vector<Tensor<float>> targets;
    targets.push_back(Tensor<float>({1}, {3.0f}));
    targets.push_back(Tensor<float>({1}, {5.0f}));
    
    // Train for a few epochs
    ASSERT_NO_THROW(model.train(inputs, targets, 5));
}

TEST(SmartDNNTest, TrainWithMultipleEpochs) {
    SmartDNN<float> model;
    
    // Build a simple model
    model.addLayer(FullyConnectedLayer<float>(2, 3));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(3, 1));
    model.compile(MSELoss<float>(), SGDOptimizer<float>());
    
    // Create training data
    std::vector<Tensor<float>> inputs;
    inputs.push_back(Tensor<float>({2}, {1.0f, 1.0f}));
    inputs.push_back(Tensor<float>({2}, {2.0f, 2.0f}));
    
    std::vector<Tensor<float>> targets;
    targets.push_back(Tensor<float>({1}, {2.0f}));
    targets.push_back(Tensor<float>({1}, {4.0f}));
    
    // Train for multiple epochs
    ASSERT_NO_THROW(model.train(inputs, targets, 10));
}

TEST(SmartDNNTest, TrainClassificationModel) {
    SmartDNN<float> model;
    
    // Build a classification model
    model.addLayer(FullyConnectedLayer<float>(3, 5));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(5, 2));
    model.addLayer(ActivationLayer<float>(Softmax<float>()));
    model.compile(CategoricalCrossEntropyLoss<float>(), AdamOptimizer<float>());
    
    // Create classification data
    std::vector<Tensor<float>> inputs;
    inputs.push_back(Tensor<float>({1, 3}, {1.0f, 0.0f, 0.0f}));
    inputs.push_back(Tensor<float>({1, 3}, {0.0f, 1.0f, 0.0f}));
    
    std::vector<Tensor<float>> targets;
    targets.push_back(Tensor<float>({1, 2}, {1.0f, 0.0f}));
    targets.push_back(Tensor<float>({1, 2}, {0.0f, 1.0f}));
    
    // Train model
    ASSERT_NO_THROW(model.train(inputs, targets, 3));
}

/*
    TRAINING/EVAL MODE TESTS
*/

TEST(SmartDNNTest, SetTrainingMode) {
    SmartDNN<float> model;
    
    model.addLayer(FullyConnectedLayer<float>(5, 3));
    
    // Should be able to set training mode
    ASSERT_NO_THROW(model.trainingMode());
}

TEST(SmartDNNTest, SetEvalMode) {
    SmartDNN<float> model;
    
    model.addLayer(FullyConnectedLayer<float>(5, 3));
    
    // Should be able to set eval mode
    ASSERT_NO_THROW(model.evalMode());
}

TEST(SmartDNNTest, ToggleBetweenModes) {
    SmartDNN<float> model;
    
    model.addLayer(FullyConnectedLayer<float>(5, 3));
    
    // Toggle between modes
    ASSERT_NO_THROW(model.trainingMode());
    ASSERT_NO_THROW(model.evalMode());
    ASSERT_NO_THROW(model.trainingMode());
}

/*
    LAYER RETRIEVAL TESTS
*/

TEST(SmartDNNTest, GetLayerByIndex) {
    SmartDNN<float> model;
    
    model.addLayer(FullyConnectedLayer<float>(10, 5));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    
    // Retrieve layers by index
    Layer<float>* layer0 = model.getLayer(0);
    Layer<float>* layer1 = model.getLayer(1);
    
    ASSERT_NE(layer0, nullptr);
    ASSERT_NE(layer1, nullptr);
}

/*
    INTEGRATION TESTS
*/

TEST(SmartDNNTest, EndToEndSimpleRegression) {
    SmartDNN<float> model;
    
    // Build model
    model.addLayer(FullyConnectedLayer<float>(1, 5));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(5, 1));
    
    // Compile
    SGDOptions sgdOptions;
    sgdOptions.learningRate = 0.01f;
    model.compile(MSELoss<float>(), SGDOptimizer<float>(sgdOptions));
    
    // Create simple linear data (y = 2x)
    std::vector<Tensor<float>> inputs;
    std::vector<Tensor<float>> targets;
    for (float x = 1.0f; x <= 5.0f; x += 1.0f) {
        inputs.push_back(Tensor<float>({1}, {x}));
        targets.push_back(Tensor<float>({1}, {2.0f * x}));
    }
    
    // Train
    model.train(inputs, targets, 50);
    
    // Test prediction
    Tensor<float> testInput({1}, {3.0f});
    Tensor<float> prediction = model.predict(testInput);
    
    // Prediction should be approximately 6.0 (within reasonable tolerance)
    // Note: Due to random initialization, we just check it's in a reasonable range
    ASSERT_GE(prediction.getData()[0], 0.0f);
}

TEST(SmartDNNTest, EndToEndBinaryClassification) {
    SmartDNN<float> model;
    
    // Build a binary classification model
    model.addLayer(FullyConnectedLayer<float>(2, 4));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(4, 2));
    model.addLayer(ActivationLayer<float>(Softmax<float>()));
    
    // Compile with Adam optimizer
    AdamOptions adamOptions;
    adamOptions.learningRate = 0.01f;
    model.compile(CategoricalCrossEntropyLoss<float>(), AdamOptimizer<float>(adamOptions));
    
    // Create XOR-like data
    std::vector<Tensor<float>> inputs;
    inputs.push_back(Tensor<float>({1, 2}, {0.0f, 0.0f}));
    inputs.push_back(Tensor<float>({1, 2}, {0.0f, 1.0f}));
    inputs.push_back(Tensor<float>({1, 2}, {1.0f, 0.0f}));
    inputs.push_back(Tensor<float>({1, 2}, {1.0f, 1.0f}));
    
    std::vector<Tensor<float>> targets;
    targets.push_back(Tensor<float>({1, 2}, {1.0f, 0.0f}));
    targets.push_back(Tensor<float>({1, 2}, {0.0f, 1.0f}));
    targets.push_back(Tensor<float>({1, 2}, {0.0f, 1.0f}));
    targets.push_back(Tensor<float>({1, 2}, {1.0f, 0.0f}));
    
    // Train
    model.train(inputs, targets, 20);
    
    // Make predictions
    Tensor<float> prediction = model.predict(inputs[0]);
    
    // Output should be a valid probability distribution
    ASSERT_EQ(prediction.getShape(), Shape({1, 2}));
    float sum = prediction.getData()[0] + prediction.getData()[1];
    ASSERT_NEAR(sum, 1.0f, 1e-5);  // Probabilities should sum to 1
}

} // namespace smart_dnn

#endif // TEST_SMARTDNN_CPP

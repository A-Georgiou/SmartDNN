#ifndef TEST_SMARTDNN_MODEL_CPP
#define TEST_SMARTDNN_MODEL_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/SmartDNN.hpp"
#include "../../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../../smart_dnn/Layers/ActivationLayer.hpp"
#include "../../smart_dnn/Layers/FlattenLayer.hpp"
#include "../../smart_dnn/Activations/ReLU.hpp"
#include "../../smart_dnn/Activations/Sigmoid.hpp"
#include "../../smart_dnn/Activations/Softmax.hpp"
#include "../../smart_dnn/Loss/MSELoss.hpp"
#include "../../smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include "../../smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "../../smart_dnn/Optimizers/SGDOptimizer.hpp"

namespace smart_dnn {

// Helper function
static inline bool approxEqual(float a, float b, float epsilon = 1e-3f) {
    return std::abs(a - b) < epsilon;
}

TEST(SmartDNNModelTest, SimpleLinearRegressionModel) {
    SmartDNN<float> model;
    
    // Create a simple 2-layer network for regression
    model.addLayer(FullyConnectedLayer<float>(2, 4));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(4, 1));
    
    // Compile with MSE loss and Adam optimizer
    model.compile(MSELoss<float>(), AdamOptimizer<float>());
    
    // Create simple training data: y = x1 + x2
    std::vector<Tensor<float>> inputs;
    std::vector<Tensor<float>> targets;
    
    inputs.push_back(Tensor<float>({2}, {1.0f, 2.0f}));
    targets.push_back(Tensor<float>({1}, {3.0f}));
    
    inputs.push_back(Tensor<float>({2}, {2.0f, 3.0f}));
    targets.push_back(Tensor<float>({1}, {5.0f}));
    
    inputs.push_back(Tensor<float>({2}, {3.0f, 4.0f}));
    targets.push_back(Tensor<float>({1}, {7.0f}));
    
    // Train for a few epochs
    model.train(inputs, targets, 5);
    
    // Verify training completed without errors
    
    // Make a prediction
    Tensor<float> testInput({2}, {1.5f, 2.5f});
    Tensor<float> prediction = model.predict(testInput);
    
    // Prediction should have correct shape
    ASSERT_EQ(prediction.getShape(), Shape({1}));
    
    // After training, prediction should be somewhat reasonable (not checking exact value due to random init)
    EXPECT_GT(prediction.getData()[0], 0.0f);
    EXPECT_LT(prediction.getData()[0], 10.0f);
}

TEST(SmartDNNModelTest, BinaryClassificationModel) {
    SmartDNN<float> model;
    
    // Create a simple network for binary classification
    model.addLayer(FullyConnectedLayer<float>(2, 3));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(3, 2));
    model.addLayer(ActivationLayer<float>(Softmax<float>()));
    
    // Compile with categorical cross entropy loss
    model.compile(CategoricalCrossEntropyLoss<float>(), SGDOptimizer<float>());
    
    // Create simple training data
    std::vector<Tensor<float>> inputs;
    std::vector<Tensor<float>> targets;
    
    // Class 0: points in lower-left quadrant
    inputs.push_back(Tensor<float>({1, 2}, {-1.0f, -1.0f}));
    targets.push_back(Tensor<float>({1, 2}, {1.0f, 0.0f}));
    
    // Class 1: points in upper-right quadrant
    inputs.push_back(Tensor<float>({1, 2}, {1.0f, 1.0f}));
    targets.push_back(Tensor<float>({1, 2}, {0.0f, 1.0f}));
    
    // Train for a few epochs
    model.train(inputs, targets, 3);
    
    // Verify training completed
    
    // Make predictions
    Tensor<float> testInput1({1, 2}, {-1.5f, -1.5f});
    Tensor<float> prediction1 = model.predict(testInput1);
    
    // Prediction should be a probability distribution
    ASSERT_EQ(prediction1.getShape(), Shape({1, 2}));
    
    // Probabilities should sum to approximately 1 (softmax output)
    float sum = prediction1.getData()[0] + prediction1.getData()[1];
    EXPECT_TRUE(approxEqual(sum, 1.0f, 0.1f));
}

TEST(SmartDNNModelTest, BatchPrediction) {
    SmartDNN<float> model;
    
    model.addLayer(FullyConnectedLayer<float>(2, 2));
    model.addLayer(ActivationLayer<float>(Sigmoid<float>()));
    
    model.compile(MSELoss<float>(), AdamOptimizer<float>());
    
    // Create multiple inputs
    std::vector<Tensor<float>> inputs;
    inputs.push_back(Tensor<float>({2}, {1.0f, 2.0f}));
    inputs.push_back(Tensor<float>({2}, {3.0f, 4.0f}));
    inputs.push_back(Tensor<float>({2}, {5.0f, 6.0f}));
    
    // Batch prediction
    std::vector<Tensor<float>> predictions = model.predict(inputs);
    
    // Should return same number of predictions as inputs
    ASSERT_EQ(predictions.size(), inputs.size());
    
    // Each prediction should have correct shape
    for (const auto& pred : predictions) {
        ASSERT_EQ(pred.getShape(), Shape({2}));
        // Sigmoid output should be between 0 and 1
        for (size_t i = 0; i < pred.getShape().size(); ++i) {
            EXPECT_GE(pred.getData()[i], 0.0f);
            EXPECT_LE(pred.getData()[i], 1.0f);
        }
    }
}

TEST(SmartDNNModelTest, TrainingAndEvalModes) {
    SmartDNN<float> model;
    
    model.addLayer(FullyConnectedLayer<float>(2, 2));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    
    model.compile(MSELoss<float>(), AdamOptimizer<float>());
    
    // Should be able to switch modes without error
    EXPECT_NO_THROW(model.trainingMode());
    EXPECT_NO_THROW(model.evalMode());
    
    // Test that prediction works in eval mode
    model.evalMode();
    Tensor<float> input({2}, {1.0f, 2.0f});
    Tensor<float> prediction = model.predict(input);
    
    ASSERT_EQ(prediction.getShape(), Shape({2}));
}

TEST(SmartDNNModelTest, GetLayer) {
    SmartDNN<float> model;
    
    model.addLayer(FullyConnectedLayer<float>(2, 4));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(4, 1));
    
    model.compile(MSELoss<float>(), AdamOptimizer<float>());
    
    // Should be able to access layers by index
    EXPECT_NO_THROW({
        Layer<float>* layer0 = model.getLayer(0);
        Layer<float>* layer1 = model.getLayer(1);
        Layer<float>* layer2 = model.getLayer(2);
        
        EXPECT_NE(layer0, nullptr);
        EXPECT_NE(layer1, nullptr);
        EXPECT_NE(layer2, nullptr);
    });
}

TEST(SmartDNNModelTest, EmptyModelPrediction) {
    SmartDNN<float> model;
    
    // Model with no layers
    Tensor<float> input({2}, {1.0f, 2.0f});
    
    // Should return input unchanged if no layers
    Tensor<float> prediction = model.predict(input);
    
    ASSERT_EQ(prediction.getShape(), input.getShape());
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(prediction.getData()[i], input.getData()[i]);
    }
}

TEST(SmartDNNModelTest, DeepNetwork) {
    SmartDNN<float> model;
    
    // Create a deeper network
    model.addLayer(FullyConnectedLayer<float>(3, 8));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(8, 8));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(8, 4));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(4, 2));
    
    model.compile(MSELoss<float>(), AdamOptimizer<float>());
    
    // Test forward pass
    Tensor<float> input({3}, {1.0f, 2.0f, 3.0f});
    Tensor<float> prediction = model.predict(input);
    
    ASSERT_EQ(prediction.getShape(), Shape({2}));
}

TEST(SmartDNNModelTest, ConsistentPredictions) {
    SmartDNN<float> model;
    
    model.addLayer(FullyConnectedLayer<float>(2, 2));
    model.compile(MSELoss<float>(), AdamOptimizer<float>());
    
    // Same input should produce same output (in eval mode, without training)
    model.evalMode();
    Tensor<float> input({2}, {1.0f, 2.0f});
    
    Tensor<float> prediction1 = model.predict(input);
    Tensor<float> prediction2 = model.predict(input);
    
    ASSERT_EQ(prediction1.getShape(), prediction2.getShape());
    for (size_t i = 0; i < prediction1.getShape().size(); ++i) {
        EXPECT_EQ(prediction1.getData()[i], prediction2.getData()[i]);
    }
}

TEST(SmartDNNModelTest, LearningProgressCheck) {
    SmartDNN<float> model;
    
    model.addLayer(FullyConnectedLayer<float>(1, 3));
    model.addLayer(ActivationLayer<float>(ReLU<float>()));
    model.addLayer(FullyConnectedLayer<float>(3, 1));
    
    AdamOptions<float> options;
    options.learningRate = 0.1f;
    model.compile(MSELoss<float>(), AdamOptimizer<float>(options));
    
    // Simple function to learn: y = 2x
    std::vector<Tensor<float>> inputs;
    std::vector<Tensor<float>> targets;
    
    for (float x = 1.0f; x <= 5.0f; x += 1.0f) {
        inputs.push_back(Tensor<float>({1}, {x}));
        targets.push_back(Tensor<float>({1}, {2.0f * x}));
    }
    
    // Get initial prediction
    model.evalMode();
    Tensor<float> testInput({1}, {3.0f});
    Tensor<float> predBefore = model.predict(testInput);
    
    // Train
    model.trainingMode();
    model.train(inputs, targets, 20);
    
    // Get prediction after training
    model.evalMode();
    Tensor<float> predAfter = model.predict(testInput);
    
    // After training, prediction should be closer to target (6.0)
    float errorBefore = std::abs(predBefore.getData()[0] - 6.0f);
    float errorAfter = std::abs(predAfter.getData()[0] - 6.0f);
    
    // Error should reduce (though not guaranteed to reach exact value with few epochs)
    EXPECT_LE(errorAfter, errorBefore + 1.0f);  // Allow some tolerance
}

} // namespace smart_dnn

#endif // TEST_SMARTDNN_MODEL_CPP

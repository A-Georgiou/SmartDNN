#include <iostream>
#include "smart_dnn/tensor/TensorBase.hpp"
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/activations/ReLU.hpp"
#include "smart_dnn/loss/MSELoss.hpp"
#include "smart_dnn/layers/FullyConnectedLayer.hpp"
#include "smart_dnn/layers/ActivationLayer.hpp"
#include "smart_dnn/datasets/SampleGenerator.hpp"
#include "smart_dnn/optimizers/AdamOptimizer.hpp"

int main() {

    using namespace sdnn;
    
    constexpr int BATCH_SIZE = 100;
    constexpr int EPOCHS = 100;
    constexpr float LEARNING_RATE = 0.01f;

    // Generate a linear dataset with 100 samples (x, y) where y = 2x + 3 + noise[0, 1]
    auto [inputs, targets] = generateLinearDataset(BATCH_SIZE);

    SmartDNN model;
    model.addLayer(new FullyConnectedLayer(1, 10));
    model.addLayer(new ActivationLayer(new ReLU()));
    model.addLayer(new FullyConnectedLayer(10, 1));

    // Compile the model with a Mean Squared Error loss function and an Adam optimizer (initialised to learning rate 0.01f).
    AdamOptions adamOptions;
    adamOptions.learningRate = LEARNING_RATE;
    model.compile(new MSELoss(), new AdamOptimizer(adamOptions));

    // Train the model on the dataset for 100 epochs.
    model.train(inputs, targets, EPOCHS);

    model.evalMode();

    // Predict the output of the model for an input of 10.0f.
    Tensor input(Shape{1}, 10.0f);

    // Print the prediction.
    Tensor prediction = model.predict(input);
    std::cout << "Input: 10.0f | Prediction: " << prediction.toString() << std::endl;
    
    return 0;
}


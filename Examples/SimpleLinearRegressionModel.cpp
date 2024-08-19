#include <iostream>
#include "../smart_dnn/Tensor.hpp"
#include "../smart_dnn/SmartDNN.hpp"
#include "../smart_dnn/Activations/ReLU.hpp"
#include "../smart_dnn/Loss/MSELoss.hpp"
#include "../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../smart_dnn/Layers/ActivationLayer.hpp"
#include "../smart_dnn/Datasets/SampleGenerator.hpp"
#include "../smart_dnn/Optimizers/AdamOptimizer.hpp"

int main() {

    // Generate a linear dataset with 100 samples (x, y) where y = 2x + 3 + noise[0, 1]
    std::pair<std::vector<Tensor>, std::vector<Tensor>> dataset = generateLinearDataset(100);

    // Initialise a SmartDNN model with a single fully connected layer and a ReLU activation function.
    // We treat the lack of output activation as a linear activation function.
    SmartDNN model;
    model.addLayer(new FullyConnectedLayer(1, 10));
    model.addLayer(new ActivationLayer(new ReLU()));
    model.addLayer(new FullyConnectedLayer(10, 1));
    
    // Compile the model with a Mean Squared Error loss function and an Adam optimizer (initialised to learning rate 0.01f).
    model.compile(new MSELoss(), new AdamOptimizer());

    // Train the model on the dataset for 100 epochs with a learning rate of 0.01f.
    model.train(dataset.first, dataset.second, 100, 0.01f);

    // Predict the output of the model for an input of 10.0f.
    Tensor input(Shape{1}, std::vector<float>{10.0f});

    // Print the prediction.
    Tensor prediction = model.predict(input);
    std::cout << "Input: 10.0f | Prediction: " << prediction << std::endl;
    
    return 0;
}


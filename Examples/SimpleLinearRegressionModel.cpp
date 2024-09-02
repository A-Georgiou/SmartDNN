#include <iostream>
#include "../smart_dnn/Tensor/Tensor.hpp"
#include "../smart_dnn/SmartDNN.hpp"
#include "../smart_dnn/Activations/ReLU.hpp"
#include "../smart_dnn/Loss/MSELoss.hpp"
#include "../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../smart_dnn/Layers/ActivationLayer.hpp"
#include "../smart_dnn/Datasets/SampleGenerator.hpp"
#include "../smart_dnn/Optimizers/AdamOptimizer.hpp"

int main() {

    using namespace smart_dnn;

    // Generate a linear dataset with 100 samples (x, y) where y = 2x + 3 + noise[0, 1]
    std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> dataset = generateLinearDataset(100);

    SmartDNN<float> model;
    model.addLayer(new FullyConnectedLayer(1, 10));
    model.addLayer(new ActivationLayer(new ReLU()));
    model.addLayer(new FullyConnectedLayer(10, 1));

    // Compile the model with a Mean Squared Error loss function and an Adam optimizer (initialised to learning rate 0.01f).
    model.compile(new MSELoss(), new AdamOptimizer());

    // Train the model on the dataset for 100 epochs with a learning rate of 0.01f.
    model.train(dataset.first, dataset.second, 100, 0.01f);

    model.evalMode();

    // Predict the output of the model for an input of 10.0f.
    Tensor input(Shape{1}, 10.0f);

    // Print the prediction.
    Tensor prediction = model.predict(input);
    std::cout << "Input: 10.0f | Prediction: " << prediction.detailedString() << std::endl;
    
    return 0;
}


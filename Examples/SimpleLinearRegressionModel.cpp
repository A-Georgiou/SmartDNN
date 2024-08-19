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

    std::pair<std::vector<Tensor>, std::vector<Tensor>> dataset = generateLinearDataset(100);

    SmartDNN model;
    model.addLayer(new FullyConnectedLayer(1, 10));
    model.addLayer(new ActivationLayer(new ReLU()));
    model.addLayer(new FullyConnectedLayer(10, 1));
    model.compile(new MSELoss(), new AdamOptimizer());
    model.train(dataset.first, dataset.second, 100, 0.01f);

    return 0;
}


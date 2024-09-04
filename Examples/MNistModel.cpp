#include "../smart_dnn/Tensor/Tensor.hpp"
#include "../smart_dnn/SmartDNN.hpp"
#include "../smart_dnn/Activations/ReLU.hpp"
#include "../smart_dnn/Activations/Softmax.hpp"
#include "../smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include "../smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "../smart_dnn/Layers/ActivationLayer.hpp"
#include "../smart_dnn/Layers/Conv2DLayer.hpp"
#include "../smart_dnn/Regularisation/MaxPooling2DLayer.hpp"
#include "../smart_dnn/Layers/FlattenLayer.hpp"
#include "../smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "../smart_dnn/Datasets/MNistLoader.hpp"
#include "../smart_dnn/TensorOperations.hpp"

int main() {

    using namespace smart_dnn;

    // Define the number of epochs and learning rate
    int epochs = 10;
    float learningRate = 0.001f;

    // Initialize the SmartDNN model
    SmartDNN<float> model;

    model.addLayer(new Conv2DLayer(1, 32, 3));                  // Conv2D layer with 32 filters, 3x3 kernel, input channels = 1
    model.addLayer(new ActivationLayer(new ReLU()));                // ReLU activation
    model.addLayer(new MaxPooling2DLayer(2));                   // Max Pooling layer with 2x2 pool size

    model.addLayer(new Conv2DLayer(32, 64, 3));                 // Conv2D layer with 64 filters, 3x3 kernel, input channels = 32
    model.addLayer(new ActivationLayer(new ReLU()));                // ReLU activation
    model.addLayer(new MaxPooling2DLayer(2));                   // Max Pooling layer with 2x2 pool size

    model.addLayer(new FlattenLayer());                         // Flatten the output of the convolutional layers

    model.addLayer(new FullyConnectedLayer(64 * 5 * 5, 128));   // Fully connected layer (5x5 spatial size after pooling)
    model.addLayer(new ActivationLayer(new ReLU()));                // ReLU activation

    model.addLayer(new FullyConnectedLayer(128, 10));           // Output layer with 10 neurons (for the 10 classes)
    model.addLayer(new ActivationLayer(new Softmax()));         // Softmax activation to produce class probabilities

    model.compile(new CategoricalCrossEntropyLoss(), new AdamOptimizer(AdamOptions{.learningRate = learningRate}));

    // Download the MNist dataset from http://yann.lecun.com/exdb/mnist/
    std::string imagesPath = ".datasets/train-images-idx3-ubyte";
    std::string labelsPath = ".datasets/train-labels-idx1-ubyte";

    // Load your MNIST dataset here
    MNISTLoader dataLoader = MNISTLoader(imagesPath, labelsPath);
    std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> dataset = dataLoader.loadData(); // Implement this method

    // Train the model on the MNIST dataset
    model.train(dataset.first, dataset.second, epochs, learningRate);

    return 0;
}
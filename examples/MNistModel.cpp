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
#include "../smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "../smart_dnn/Regularisation/DropoutLayer.hpp"

int main() {

    using namespace smart_dnn;

    // Define the number of epochs and learning rate
    constexpr int EPOCHS = 10;
    constexpr int BATCH_SIZE = 8;
    constexpr int SAMPLE_COUNT = 1000;
    constexpr float LEARNING_RATE = 0.001f;

    // Initialize the SmartDNN MNist model
    SmartDNN<float> model;

    model.addLayer(new Conv2DLayer(1, 32, 3));           // Conv2D layer
    model.addLayer(new BatchNormalizationLayer(32));     // Batch normalization after conv
    model.addLayer(new ActivationLayer(new ReLU()));     // ReLU activation
    model.addLayer(new MaxPooling2DLayer(2, 2));         // Added MaxPooling
    model.addLayer(new DropoutLayer(0.25f));              // Reduced dropout rate

    model.addLayer(new FlattenLayer());                  // Flatten layer
    model.addLayer(new FullyConnectedLayer(5408, 128));  // Adjusted input size due to MaxPooling
    model.addLayer(new BatchNormalizationLayer(128));    // Batch normalization after FC
    model.addLayer(new ActivationLayer(new ReLU()));     // ReLU activation
    model.addLayer(new DropoutLayer(0.25f));              // Reduced dropout rate

    model.addLayer(new FullyConnectedLayer(128, 10));    // Output layer
    model.addLayer(new ActivationLayer(new Softmax()));  // Softmax activation

    AdamOptions adamOptions;
    adamOptions.learningRate = LEARNING_RATE;
    adamOptions.beta1 = 0.9f;
    adamOptions.beta2 = 0.999f;
    adamOptions.epsilon = 1e-8f;
    adamOptions.l1Strength = 0.0f; 
    adamOptions.l2Strength = 0.0f;  
    adamOptions.decay = 0.0f;  
    model.compile(new CategoricalCrossEntropyLoss(), new AdamOptimizer(adamOptions));

    // Download the MNist dataset from http://yann.lecun.com/exdb/mnist/
    std::string imagesPath = ".datasets/train-images-idx3-ubyte";
    std::string labelsPath = ".datasets/train-labels-idx1-ubyte";

    // Load your MNIST dataset here
    MNISTLoader dataLoader = MNISTLoader(imagesPath, labelsPath, BATCH_SIZE, SAMPLE_COUNT);
    std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> dataset = dataLoader.loadData(); // Implement this method

    // Train the model on the MNIST dataset
    model.train(dataset.first, dataset.second, EPOCHS);

    model.evalMode();     

    // Load your MNIST dataset here
    MNISTLoader dataLoaderTest = MNISTLoader(imagesPath, labelsPath, BATCH_SIZE);
    std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> testDataset = dataLoaderTest.loadData(); // Implement this method

    for (int i = testDataset.first.size()-1; i > testDataset.first.size()-5; i--){
        Tensor<float> input = testDataset.first[i];
        Tensor<float> output = testDataset.second[i];

        Tensor prediction = model.predict(input);
        std::cout << "Input: " << dataLoader.toAsciiArt(input) << std::endl;
        std::cout << "Prediction: " << prediction.toDataString() << std::endl;
        std::cout << "Actual value: " << output.toDataString() << std::endl;
    }

    return 0;
}
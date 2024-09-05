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
#include "../smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "../smart_dnn/Regularisation/DropoutLayer.hpp"

int main() {

    using namespace smart_dnn;

    // Define the number of epochs and learning rate
    int epochs = 10;
    float learningRate = 0.001f;

   // Initialize the SmartDNN model
    SmartDNN<float> model;

    model.addLayer(new Conv2DLayer<float>(1, 32, 3));  // Conv2D layer
    model.addLayer(new BatchNormalizationLayer<float>(32));  // Batch normalization after conv
    model.addLayer(new ActivationLayer<float>(new ReLU()));  // ReLU activation
    model.addLayer(new MaxPooling2DLayer<float>(2, 2));  // Added MaxPooling
    model.addLayer(new DropoutLayer<float>(0.25));  // Reduced dropout rate

    model.addLayer(new FlattenLayer<float>());
    model.addLayer(new FullyConnectedLayer<float>(5408, 128));  // Adjusted input size due to MaxPooling
    model.addLayer(new BatchNormalizationLayer<float>(128));
    model.addLayer(new ActivationLayer<float>(new ReLU()));
    model.addLayer(new DropoutLayer<float>(0.25));  // Reduced dropout rate

    model.addLayer(new FullyConnectedLayer<float>(128, 10));
    model.addLayer(new ActivationLayer<float>(new Softmax()));

    AdamOptions<float> adamOptions;
    adamOptions.learningRate = learningRate;
    adamOptions.beta1 = 0.9f;
    adamOptions.beta2 = 0.999f;
    adamOptions.epsilon = 1e-8f;
    adamOptions.l1Strength = 0.0f;
    adamOptions.l2Strength = 0.0f;  // Removed L2 regularization
    adamOptions.decay = 0.0f;  // Removed learning rate decay
    model.compile(new CategoricalCrossEntropyLoss(), new AdamOptimizer(adamOptions));

    // Download the MNist dataset from http://yann.lecun.com/exdb/mnist/
    std::string imagesPath = ".datasets/train-images-idx3-ubyte";
    std::string labelsPath = ".datasets/train-labels-idx1-ubyte";

    // Load your MNIST dataset here
    MNISTLoader dataLoader = MNISTLoader(imagesPath, labelsPath, 1, 5);
    std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> dataset = dataLoader.loadData(); // Implement this method

    for (auto& input : dataset.first) {
        input = input / 255.0f;  // Normalize to [0, 1] range
    }

    // Train the model on the MNIST dataset
    model.train(dataset.first, dataset.second, epochs, learningRate);

    model.evalMode();     

    // Load your MNIST dataset here
    MNISTLoader dataLoaderTest = MNISTLoader(imagesPath, labelsPath, 1);
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
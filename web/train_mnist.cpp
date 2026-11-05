// Trains the demo model on all 60k MNIST images and saves it: c++ -std=c++17 -O3 -I . web/train_mnist.cpp -o train_mnist && ./train_mnist .datasets web/model.bin 5
#include "mnist_model.hpp"
#include "smart_dnn/Datasets/MNistLoader.hpp"
#include "smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace smart_dnn;

int main(int argc, char** argv) {
    const std::string datasets = argc > 1 ? argv[1] : ".datasets";
    const std::string output = argc > 2 ? argv[2] : "web/model.bin";
    const int epochs = argc > 3 ? std::atoi(argv[3]) : 5;
    constexpr int BATCH_SIZE = 128;

    SmartDNN<float> model;
    addMnistLayers(model);
    model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer(AdamOptions<float>()));

    auto [trainInputs, trainTargets] = MNISTLoader(datasets + "/train-images-idx3-ubyte", datasets + "/train-labels-idx1-ubyte", BATCH_SIZE).loadData();
    model.train(trainInputs, trainTargets, epochs);

    model.evalMode();
    auto [testInputs, testTargets] = MNISTLoader(datasets + "/t10k-images-idx3-ubyte", datasets + "/t10k-labels-idx1-ubyte", BATCH_SIZE).loadData();
    int correct = 0, total = 0;
    for (size_t b = 0; b < testInputs.size(); ++b) {
        Tensor<float> prediction = model.predict(testInputs[b]);
        for (int row = 0; row < prediction.getShape()[0]; ++row, ++total) {
            const float* p = &prediction[static_cast<size_t>(row) * 10];
            const float* y = &testTargets[b][static_cast<size_t>(row) * 10];
            correct += (std::max_element(p, p + 10) - p) == (std::max_element(y, y + 10) - y);
        }
    }
    std::printf("Test accuracy: %.2f%% (%d/%d)\n", 100.0 * correct / total, correct, total);

    model.saveModel(output);
    std::printf("Saved %s\n", output.c_str());
    return 0;
}

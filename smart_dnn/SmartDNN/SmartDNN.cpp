#include "../SmartDNN.hpp"
#include "../Debugging/Logger.hpp"
#include "../TensorOperations.hpp"
#include <chrono>

SmartDNN::SmartDNN() {
    lossFunction = nullptr;
    optimizer = nullptr;
}

SmartDNN::~SmartDNN() {
    for (Layer* layer : layers) {
        delete layer;
    }

    delete lossFunction;
    delete optimizer;
}

void SmartDNN::addLayer(Layer* layer) {
    layers.push_back(layer);
}

void SmartDNN::compile(Loss* loss, Optimizer* optimizer) {
    this->lossFunction = loss;
    this->optimizer = optimizer;
}

void SmartDNN::train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int epochs, float learningRate) {
    std::cout << "Begin training" << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;

        for (size_t i = 0; i < inputs.size(); ++i) {
            Logger::log(Logger::Level::INFO, "Training on sample " + std::to_string(i));
            Tensor prediction = inputs[i];
            for (Layer* layer : layers) {
                prediction = layer->forward(prediction);
            }

            totalLoss += lossFunction->compute(prediction, targets[i]);

            Tensor gradOutput = lossFunction->gradient(prediction, targets[i]);

            for (int j = layers.size() - 1; j >= 0; --j) {
                gradOutput = layers[j]->backward(gradOutput);
            }

            for (Layer* layer : layers) {
                layer->updateWeights(*optimizer);
            }
        }
        std::cout << "Epoch " << epoch << " - Loss: " << totalLoss / inputs.size() << std::endl;
    }
}

Tensor SmartDNN::predict(const Tensor& input) {
    Tensor prediction = input;
    for (Layer* layer : layers) {
        prediction = layer->forward(prediction);
    }
    return prediction;
}

std::vector<Tensor> SmartDNN::predict(const std::vector<Tensor>& inputs) {
    std::vector<Tensor> predictions;
    for (const Tensor& input : inputs) {
        predictions.push_back(predict(input));
    }
    return predictions;
}

void SmartDNN::trainingMode() {
    setTrainingMode(true);
}

void SmartDNN::evalMode() {
    setTrainingMode(false);
}

void SmartDNN::setTrainingMode(bool trainingMode) {
    for (Layer* layer : layers) {
        layer->setTrainingMode(true);
    }
}
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
        std::cout << "Epoch " << epoch << std::endl;
        float totalLoss = 0.0f;

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::cout << "Training on sample " << i << std::endl;
            Tensor prediction = inputs[i];
            for (Layer* layer : layers) {
                std::cout << "Forwarding through layer" << std::endl;
                prediction = layer->forward(prediction);
            }

            std::cout << "Computing loss" << std::endl;
            totalLoss += lossFunction->compute(prediction, targets[i]);

            std::cout << "Computing gradient" << std::endl;
            Tensor gradOutput = lossFunction->gradient(prediction, targets[i]);

            for (int j = layers.size() - 1; j >= 0; --j) {
                std::cout << "Backwarding through layer" << std::endl;
                gradOutput = layers[j]->backward(gradOutput);
            }

            for (Layer* layer : layers) {
                std::cout << "Updating weights" << std::endl;
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
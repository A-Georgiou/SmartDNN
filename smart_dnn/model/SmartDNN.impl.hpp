#ifndef SMART_DNN_IMPL_HPP
#define SMART_DNN_IMPL_HPP

#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/debugging/Logger.hpp"

namespace sdnn {

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

void SmartDNN::backward(const Tensor& gradOutput)  {
    Tensor grad = gradOutput;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad); 
    }
}

void SmartDNN::updateWeights(Optimizer& optimizer) {
        for (auto& layer : layers) {
            layer->updateWeights(optimizer);  // Assuming each layer has its own updateWeights method
        }
    }


void SmartDNN::compile(Loss* loss, Optimizer* optimizer) {
    this->lossFunction = loss;
    this->optimizer = optimizer;
}

Layer* SmartDNN::getLayer(size_t index) const {
    return layers[index];
}

void SmartDNN::train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Tensor totalLoss = zeros({1}, dtype::f32);
        size_t inputSize = inputs.size();

        for (size_t i = 0; i < inputSize; ++i) {

            Tensor prediction = inputs[i];
            for (Layer* layer : layers) {
                prediction = layer->forward(prediction);
            }

            totalLoss += lossFunction->compute(prediction, targets[i]);
            Tensor gradOutput = lossFunction->gradient(prediction, targets[i]);

            for (int j = layers.size() - 1; j >= 0; --j) {
                gradOutput = layers[j]->backward(gradOutput);
                layers[j]->updateWeights(*optimizer);
            }

        }
        std::cout << "Epoch " << epoch << " - Loss: " << (totalLoss / (inputs.size())).toDataString() << std::endl;
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
        layer->setTrainingMode(trainingMode);
    }
}

} // namespace sdnn


#endif // SMART_DNN_IMPL_HPP
#ifndef SMART_DNN_IMPL_HPP
#define SMART_DNN_IMPL_HPP

#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/debugging/Logger.hpp"

namespace smart_dnn {

template <typename T>
void SmartDNN<T>::backward(const Tensor<T>& gradOutput) {
    Tensor<T> grad = gradOutput;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad); 
    }
}

template <typename T>
void SmartDNN<T>::updateWeights() {
    for (auto& layer : layers) {
        layer->updateWeights(*optimizer);
    }
}

template <typename T>
Layer<T>* SmartDNN<T>::getLayer(size_t index) const {
    return layers[index].get();
}

template <typename T>
void SmartDNN<T>::train(const std::vector<Tensor<T>>& inputs, const std::vector<Tensor<T>>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Tensor<T> totalLoss{Shape({1}), T(0)};
        size_t inputSize = inputs.size();

        for (size_t i = 0; i < inputSize; ++i) {
            Tensor<T> prediction = inputs[i];
            for (const auto& layer : layers) {
                prediction = layer->forward(prediction);
            }

            totalLoss += lossFunction->compute(prediction, targets[i]);
            Tensor<T> gradOutput = lossFunction->gradient(prediction, targets[i]);

            backward(gradOutput);
            updateWeights();
        }
        std::cout << "Epoch " << epoch << " - Loss: " << (totalLoss / T(inputs.size())).toDataString() << std::endl;
    }
}

template <typename T>
Tensor<T> SmartDNN<T>::predict(const Tensor<T>& input) {
    Tensor<T> prediction = input;
    for (const auto& layer : layers) {
        prediction = layer->forward(prediction);
    }
    return prediction;
}

template <typename T>
std::vector<Tensor<T>> SmartDNN<T>::predict(const std::vector<Tensor<T>>& inputs) {
    std::vector<Tensor<T>> predictions;
    predictions.reserve(inputs.size());
    for (const Tensor<T>& input : inputs) {
        predictions.push_back(predict(input));
    }
    return predictions;
}

template <typename T>
void SmartDNN<T>::trainingMode() {
    setTrainingMode(true);
}

template <typename T>
void SmartDNN<T>::evalMode() {
    setTrainingMode(false);
}

template <typename T>
void SmartDNN<T>::setTrainingMode(bool trainingMode) {
    for (const auto& layer : layers) {
        layer->setTrainingMode(trainingMode);
    }
}

} // namespace smart_dnn


#endif // SMART_DNN_IMPL_HPP
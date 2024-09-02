#ifndef SMART_DNN_IMPL_HPP
#define SMART_DNN_IMPL_HPP

#include "../SmartDNN.hpp"
#include "../Debugging/Logger.hpp"
#include "../TensorOperations.hpp"

namespace smart_dnn {

template <typename T>
SmartDNN<T>::SmartDNN() {
    lossFunction = nullptr;
    optimizer = nullptr;
}

template <typename T>
SmartDNN<T>::~SmartDNN() {
    for (Layer<T>* layer : layers) {
        delete layer;
    }

    delete lossFunction;
    delete optimizer;
}

template <typename T>
void SmartDNN<T>::addLayer(Layer<T>* layer) {
    layers.push_back(layer);
}

template <typename T>
void SmartDNN<T>::compile(Loss<T>* loss, Optimizer<T>* optimizer) {
    this->lossFunction = loss;
    this->optimizer = optimizer;
}

template <typename T>
void SmartDNN<T>::train(const std::vector<Tensor<T>>& inputs, const std::vector<Tensor<T>>& targets, int epochs, float learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Tensor<T> totalLoss{Shape({1}), T(0)};

        for (size_t i = 0; i < inputs.size(); ++i) {
            Tensor<T> prediction = inputs[i];
            for (Layer<T>* layer : layers) {
                prediction = layer->forward(prediction);
            }

            totalLoss += lossFunction->compute(prediction, targets[i]);
            Tensor<T> gradOutput = lossFunction->gradient(prediction, targets[i]);

            for (int j = layers.size() - 1; j >= 0; --j) {
                gradOutput = layers[j]->backward(gradOutput);
            }

            for (Layer<T>* layer : layers) {
                layer->updateWeights(*optimizer);
            }
        }
        std::cout << "Epoch " << epoch << " - Loss: " << (totalLoss / T(inputs.size())).toDataString() << std::endl;
    }
}

template <typename T>
Tensor<T> SmartDNN<T>::predict(const Tensor<T>& input) {
    Tensor<T> prediction = input;
    for (Layer<T>* layer : layers) {
        prediction = layer->forward(prediction);
    }
    return prediction;
}

template <typename T>
std::vector<Tensor<T>> SmartDNN<T>::predict(const std::vector<Tensor<T>>& inputs) {
    std::vector<Tensor<T>> predictions;
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
    for (Layer<T>* layer : layers) {
        layer->setTrainingMode(true);
    }
}

}


#endif // SMART_DNN_IMPL_HPP
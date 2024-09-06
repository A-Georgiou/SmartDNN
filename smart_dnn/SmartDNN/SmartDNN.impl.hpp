#ifndef SMART_DNN_IMPL_HPP
#define SMART_DNN_IMPL_HPP

#include "../SmartDNN.hpp"
#include "../Debugging/Logger.hpp"

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
void SmartDNN<T>::backward(const Tensor<T>& gradOutput)  {
    Tensor<T> grad = gradOutput;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad); 
    }
}

template <typename T>
void SmartDNN<T>::updateWeights(Optimizer<T>& optimizer) {
        for (auto& layer : layers) {
            layer->updateWeights(optimizer);  // Assuming each layer has its own updateWeights method
        }
    }


template <typename T>
void SmartDNN<T>::compile(Loss<T>* loss, Optimizer<T>* optimizer) {
    this->lossFunction = loss;
    this->optimizer = optimizer;
}

template <typename T>
Layer<T>* SmartDNN<T>::getLayer(size_t index) const {
    return layers[index];
}

template <typename T>
void SmartDNN<T>::train(const std::vector<Tensor<T>>& inputs, const std::vector<Tensor<T>>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Tensor<T> totalLoss{Shape({1}), T(0)};
        size_t inputSize = inputs.size();

        for (size_t i = 0; i < inputSize; ++i) {

            Tensor<T> prediction = inputs[i];
            for (Layer<T>* layer : layers) {
                prediction = layer->forward(prediction);
            }

            totalLoss += lossFunction->compute(prediction, targets[i]);
            Tensor<T> gradOutput = lossFunction->gradient(prediction, targets[i]);

            for (int j = layers.size() - 1; j >= 0; --j) {
                gradOutput = layers[j]->backward(gradOutput);
                layers[j]->updateWeights(*optimizer);
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
        layer->setTrainingMode(trainingMode);
    }
}

} // namespace smart_dnn


#endif // SMART_DNN_IMPL_HPP
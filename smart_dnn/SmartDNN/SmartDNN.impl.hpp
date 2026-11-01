#ifndef SMART_DNN_IMPL_HPP
#define SMART_DNN_IMPL_HPP

#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Debugging/Logger.hpp"

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
void SmartDNN<T>::saveModel(const std::string& filename) const {
    std::ofstream os(filename, std::ios::binary);
    if (!os) {
        throw std::runtime_error("Cannot open " + filename + " for writing");
    }
    const uint32_t header[4] = {modelMagic, modelVersion, static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(layers.size())};
    os.write(reinterpret_cast<const char*>(header), sizeof(header));
    for (const auto& layer : layers) {
        layer->save(os);
    }
    if (!os) {
        throw std::runtime_error("Failed to write " + filename);
    }
}

template <typename T>
void SmartDNN<T>::loadModel(const std::string& filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is) {
        throw std::runtime_error("Cannot open " + filename);
    }
    uint32_t header[4] = {};
    is.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!is || header[0] != modelMagic || header[1] != modelVersion || header[2] != sizeof(T) || header[3] != layers.size()) {
        throw std::runtime_error(filename + " is not a SmartDNN model with this model's layer count and element type");
    }
    for (auto& layer : layers) {
        layer->load(is);
    }
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
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
void SmartDNN<T>::updateWeights(T learningRate) {
    std::vector<std::reference_wrapper<Tensor<T>>> values;
    std::vector<std::reference_wrapper<Tensor<T>>> gradients;
    for (const auto& layer : layers) {
        for (const Parameter<T>& parameter : layer->parameters()) {
            if (!parameter.gradient.get()) {
                throw std::logic_error("updateWeights needs a backward pass first");
            }
            values.push_back(parameter.value);
            gradients.push_back(*parameter.gradient.get());
        }
    }
    if (values.empty()) {
        return;
    }

    if (maxGradientNorm > T(0)) {
        clipGradients(gradients);
    }

    // One call per update, so optimizers that count steps (Adam) advance once per training step.
    optimizer->optimize(values, gradients, learningRate);
}

template <typename T>
void SmartDNN<T>::clipGradients(const std::vector<std::reference_wrapper<Tensor<T>>>& gradients) const {
    // Dividing by the largest magnitude before squaring keeps exploding gradients from overflowing the norm.
    T largest = T(0);
    for (const Tensor<T>& gradient : gradients) {
        for (T value : gradient.getData()) {
            largest = std::max(largest, std::abs(value));
        }
    }
    if (largest == T(0)) {
        return;
    }

    T scaledSquares = T(0);
    for (const Tensor<T>& gradient : gradients) {
        for (T value : gradient.getData()) {
            scaledSquares += (value / largest) * (value / largest);
        }
    }
    const T norm = largest * std::sqrt(scaledSquares);
    if (norm > maxGradientNorm) {
        for (Tensor<T>& gradient : gradients) {
            gradient = gradient * (maxGradientNorm / norm);
        }
    }
}

template <typename T>
void SmartDNN<T>::setMaxGradientNorm(T maxNorm) {
    if (!std::isfinite(maxNorm) || maxNorm < T(0)) {
        throw std::invalid_argument("maxNorm must be finite and not negative");
    }
    maxGradientNorm = maxNorm;
}

template <typename T>
Layer<T>* SmartDNN<T>::getLayer(size_t index) const {
    return layers[index].get();
}

template <typename T>
size_t SmartDNN<T>::getLayerCount() const {
    return layers.size();
}

template <typename T>
void SmartDNN<T>::saveModel(const std::string& filename, WeightFormat format) const {
    std::ofstream os(filename, std::ios::binary);
    if (!os) {
        throw std::runtime_error("Cannot open " + filename + " for writing");
    }
    const uint32_t bytesPerWeight = format == WeightFormat::Int8 ? 1 : static_cast<uint32_t>(sizeof(T));
    const uint32_t header[4] = {modelMagic, modelVersion, bytesPerWeight, static_cast<uint32_t>(layers.size())};
    os.write(reinterpret_cast<const char*>(header), sizeof(header));
    int8TensorStorage(os) = format == WeightFormat::Int8;
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
    if (!is || header[0] != modelMagic || header[1] != modelVersion || (header[2] != sizeof(T) && header[2] != 1) || header[3] != layers.size()) {
        throw std::runtime_error(filename + " is not a SmartDNN model with this model's layer count and element type");
    }
    int8TensorStorage(is) = header[2] == 1;
    for (auto& layer : layers) {
        layer->load(is);
    }
}

template <typename T>
void SmartDNN<T>::train(const std::vector<Tensor<T>>& inputs, const std::vector<Tensor<T>>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Tensor<T> totalLoss{Shape({1}), T(0)};
        for (size_t i = 0; i < inputs.size(); ++i) {
            totalLoss[0] += trainStep(inputs[i], targets[i]);
        }
        std::cout << "Epoch " << epoch << " - Loss: " << (totalLoss / T(inputs.size())).toDataString() << std::endl;
    }
}

template <typename T>
T SmartDNN<T>::trainStep(const Tensor<T>& input, const Tensor<T>& target, T learningRate) {
    Tensor<T> prediction = predict(input);
    T loss = lossFunction->compute(prediction, target)[0];
    backward(lossFunction->gradient(prediction, target));
    updateWeights(learningRate);
    return loss;
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
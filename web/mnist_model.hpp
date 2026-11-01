#ifndef MNIST_MODEL_HPP
#define MNIST_MODEL_HPP

#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Activations/Softmax.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Layers/Conv2DLayer.hpp"
#include "smart_dnn/Layers/FlattenLayer.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "smart_dnn/Regularisation/DropoutLayer.hpp"
#include "smart_dnn/Regularisation/MaxPooling2DLayer.hpp"

namespace smart_dnn {

// The examples/MNistModel.cpp network; training and the browser build must add identical layers.
inline void addMnistLayers(SmartDNN<float>& model) {
    model.addLayer(Conv2DLayer(1, 32, 3));
    model.addLayer(BatchNormalizationLayer(32));
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(MaxPooling2DLayer(2, 2));
    model.addLayer(DropoutLayer(0.25f));

    model.addLayer(FlattenLayer());
    model.addLayer(FullyConnectedLayer(5408, 128));
    model.addLayer(BatchNormalizationLayer(128));
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(DropoutLayer(0.25f));

    model.addLayer(FullyConnectedLayer(128, 10));
    model.addLayer(ActivationLayer(Softmax()));
}

} // namespace smart_dnn

#endif // MNIST_MODEL_HPP

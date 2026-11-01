// WebAssembly entry point for the MNIST demo: loads the embedded trained model and classifies 28x28 drawings.
#include "mnist_model.hpp"
#include <emscripten/emscripten.h>
#include <algorithm>

using namespace smart_dnn;

namespace {
SmartDNN<float> model;
float probabilities[10];
}

extern "C" {

// Builds the network and loads /model.bin, which web/build.sh embeds into the module.
EMSCRIPTEN_KEEPALIVE void load_model() {
    addMnistLayers(model);
    model.loadModel("/model.bin");
    model.evalMode();
}

// Classifies 784 row-major pixels in [0, 1] (white digit on black) and returns 10 class probabilities.
EMSCRIPTEN_KEEPALIVE const float* predict(const float* pixels) {
    const Tensor<float> output = model.predict(Tensor<float>(Shape({1, 1, 28, 28}), pixels));
    std::copy(&output[0], &output[0] + 10, probabilities);
    return probabilities;
}

}

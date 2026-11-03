# SmartDNN

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/language-C%2B%2B-orange.svg)](https://isocpp.org/)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/a-georgiou/SmartDNN)

A high-performance C++ deep learning library designed for flexibility and efficiency.

## Overview

SmartDNN is a modern C++ deep learning framework that offers a clean, intuitive API for building and training neural networks while maintaining C++'s performance advantages. The library focuses on providing a high-level interface that simplifies neural network development without sacrificing computational efficiency.

### Key Features

- **Flexible Architecture**: Easily build and customize neural network architectures
- **C++17 Implementation**: Templated tensor operations with CPU storage
- **Comprehensive Layer Support**: Full suite of essential neural network layers
- **Customizable Training**: Multiple loss functions and optimization methods
- **Clean API**: Intuitive interface for model building and training

## Performance Highlights

The numbers below are historical reports, not a reproducible current benchmark.
Their original machine, compiler and measurement date were not recorded here.
The 2026-10-29 e2e audit at `bce6ecb` could not compile either example on
case-sensitive Linux, so it provides no current training or performance baseline.
Do not treat these figures as verified for the current code.

### Linear Regression Model (1000 samples, 1000 epochs)
- **Non-templated runtime**: ~17680ms
- **Optimized templated runtime**: ~8325ms
- **Performance gain**: ~53% improvement

### MNIST Classification (1000 samples, batch size: 64, 1000 epochs)
- **Non-templated runtime**: ~83 minutes per epoch
- **Optimized templated runtime**: ~10969ms per epoch
- **Performance gain**: ~99.8% improvement

## Getting Started

The following is a model-building fragment, not a standalone program. It assumes
`using namespace smart_dnn;`, the corresponding public headers, and caller-defined
`std::vector<Tensor<float>> inputs, targets` and an integer `epochs`.
See [example models](#example-models) for sources and current build limitations.

```cpp
// Initialize the model
SmartDNN model;

// Define architecture
model.addLayer(FullyConnectedLayer(10, 100));        // Input -> Hidden
model.addLayer(ActivationLayer(ReLU()));             // ReLU activation
model.addLayer(FullyConnectedLayer(100, 100));       // Hidden -> Hidden
model.addLayer(ActivationLayer(Sigmoid()));          // Sigmoid activation
model.addLayer(FullyConnectedLayer(100, 10));        // Hidden -> Output
model.addLayer(ActivationLayer(Softmax()));          // Softmax for classification

// Compile and train
model.compile(MSELoss(), AdamOptimizer());
model.train(inputs, targets, epochs);
```

## Advanced Example: MNIST CNN Model

This fragment has the same namespace and data requirements as the first example
and additionally assumes a float `learningRate`. Image batches use shape
`{batch, 1, 28, 28}` and one-hot targets use `{batch, 10}`.
Call `model.evalMode()` before inference with dropout or batch normalization;
`predict` does not switch modes automatically.

```cpp
// Initialize the SmartDNN MNIST model
SmartDNN<float> model;

// Convolutional layers
model.addLayer(Conv2DLayer(1, 32, 3));               // Conv2D layer
model.addLayer(BatchNormalizationLayer(32));         // Batch normalization
model.addLayer(ActivationLayer(ReLU()));             // ReLU activation
model.addLayer(MaxPooling2DLayer(2, 2));             // MaxPooling
model.addLayer(DropoutLayer(0.25f));                 // Dropout for regularization

// Fully connected layers
model.addLayer(FlattenLayer());                      // Flatten layer
model.addLayer(FullyConnectedLayer(5408, 128));      // FC layer
model.addLayer(BatchNormalizationLayer(128));        // Batch normalization
model.addLayer(ActivationLayer(ReLU()));             // ReLU activation
model.addLayer(DropoutLayer(0.25f));                 // Dropout

// Output layer
model.addLayer(FullyConnectedLayer(128, 10));        // Output layer
model.addLayer(ActivationLayer(Softmax()));          // Softmax activation

// Configure optimizer options
AdamOptions adamOptions;
adamOptions.learningRate = learningRate;
adamOptions.beta1 = 0.9f;
adamOptions.beta2 = 0.999f;
adamOptions.epsilon = 1e-8f;

// Compile and train
model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer(adamOptions));
model.train(inputs, targets, epochs);
```

## Components

### Available Layers
- **Fully Connected Layer**: Dense neural network layers
- **Convolutional 2D Layer**: For image processing tasks
- **Activation Layers**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Swish, Mish
- **Regularization Layers**: Dropout, Batch Normalization
- **Pooling Layers**: Max Pooling 2D
- **Utility Layers**: Flatten

### Optimizers
- **Adam**: Adaptive Moment Estimation optimizer with configurable parameters
- **SGD**: Momentum, dampening, weight decay and optional Nesterov momentum
- **RMSProp**: Moving squared-gradient average, optional momentum and centered mode
- **AdaGrad**: Cumulative squared gradients with configurable learning rate and epsilon

Optimizer headers live in `smart_dnn/Optimizers/`. For example, this standalone
program applies one AdaGrad update:

```cpp
#include <functional>
#include <memory>
#include <vector>
#include "smart_dnn/Optimizers/AdaGradOptimizer.hpp"

int main() {
    using namespace smart_dnn;
    Tensor<float> weight(Shape({2}), 1.0f);
    Tensor<float> gradient(Shape({2}), 0.5f);
    AdaGradOptions<float> options;
    options.learningRate = 0.01f;
    AdaGradOptimizer<float> optimizer(options);
    std::vector<std::reference_wrapper<Tensor<float>>> weights{weight};
    std::vector<std::reference_wrapper<Tensor<float>>> gradients{gradient};
    optimizer.optimize(weights, gradients);
}
```

`optimize(weights, gradients, learningRateOverride)` uses an override only when
positive. AdaGrad rejects unequal vector lengths or paired tensor element counts.
Its state persists per Tensor object address, so keep parameter objects alive
and stable while reusing the optimizer.

### Loss Functions
- **Mean Squared Error (MSE)**: For regression tasks
- **Categorical Cross Entropy**: For classification tasks

## Performance Optimizations

SmartDNN includes several key optimizations:

- **Slice View**: Access tensor data without copying
- **Broadcast View**: Efficient data broadcasting for better performance
- **Transforms**: Iterator-based transforms for compiler optimizations
- **Clean Architecture**: Single responsibility principle for better code organization
- **Template Specialization**: Type-specific optimizations
- **Parallel Directives**: Random tensor generation contains an OpenMP pragma; the supplied CMake projects do not enable OpenMP

## Installation

### Standard Installation

Requirements: a C++17 compiler and CMake 3.20 or newer for the commands below
(the project declares a 3.10 minimum). The root CMake project
builds an application named `SmartDNN`, not an installable library target.
Public headers are included relative to the repository root.

1. Clone the repository:
   ```bash
   git clone https://github.com/A-Georgiou/SmartDNN.git
   ```

2. Enter the checkout and create `src/main.cpp` with your application code
   **before** configuring CMake:
   ```bash
   cd SmartDNN
   mkdir -p src
   ```

3. Configure and build your application:
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j2
   ```

4. Run the program:
   ```bash
   ./build/SmartDNN
   ```

### Unit Tests

The tests are a separate CMake project and do not require `src/main.cpp`.
Configuration downloads GoogleTest, so network access is required.
From the repository root:

```bash
cmake -S tests -B build-tests -DCMAKE_BUILD_TYPE=Release
cmake --build build-tests -j2
ctest --test-dir build-tests --output-on-failure
```

CTest registers one test named `TensorTests` that runs the GoogleTest suite.
CI also builds this project and runs `RunTests` on main pushes and pull requests,
or through manual dispatch. This does not validate example training.

### Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/A-Georgiou/SmartDNN.git
   ```

2. Enter the checkout and create `src/main.cpp` before building the image.
   The same example build limitations below apply inside Docker.

3. Build the Docker image:
   ```bash
   docker build -f .docker/Dockerfile -t smartdnn-app .
   ```

4. Run the project:
   ```bash
   docker run --rm -it smartdnn-app
   ```

## Example Models

- [Simple Linear Regression model](examples/SimpleLinearRegressionModel.cpp)
- [CNN for MNIST Classification](examples/MNistModel.cpp)

Known limitation at `bce6ecb` (2026-10-29): the linear example includes lowercase
module paths, and `SmartDNN.impl.hpp` includes `smart_dnn/debugging/Logger.hpp`
instead of the existing `smart_dnn/Debugging/Logger.hpp`. Both examples fail to
compile on case-sensitive Linux, including model fragments using `SmartDNN.hpp`;
see [#103](https://github.com/A-Georgiou/SmartDNN/issues/103).
The standalone AdaGrad example above does not include that model header.
MNIST additionally expects uncompressed IDX image and label files at the paths
specified in its source. No current training accuracy is established.

Current tensor storage is CPU-only. GPU marker types do not provide a GPU backend,
and model `saveModel`/`loadModel` declarations have no implementation.
See [CHANGELOG.md](CHANGELOG.md) for merged changes.

## Future Roadmap

- **Extended Layer Support**: Additional layer types including advanced convolutional and recurrent layers
- **Advanced Network Architectures**: More flexible and customizable network structures
- **GPU Acceleration**: CUDA integration for GPU-based training and inference
- **Comprehensive Documentation**: Detailed guides and examples

## Contributing

Contributions are welcome! If you would like to contribute to the project, please reach out via the contact information below.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or inquiries, please contact [AndrewGeorgiou98@outlook.com](mailto:andrewgeorgiou98@outlook.com).

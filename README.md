# SmartDNN

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/language-C%2B%2B-orange.svg)](https://isocpp.org/)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/a-georgiou/SmartDNN)

A high-performance C++ deep learning library designed for flexibility and efficiency.

## Overview

SmartDNN is a modern C++ deep learning framework that offers a clean, intuitive API for building and training neural networks while maintaining C++'s performance advantages. The library focuses on providing a high-level interface that simplifies neural network development without sacrificing computational efficiency.

### Key Features

- **Flexible Architecture**: Easily build and customize neural network architectures
- **High Performance**: Optimized C++ implementation with significant runtime improvements
- **Comprehensive Layer Support**: Full suite of essential neural network layers
- **Customizable Training**: Multiple loss functions and optimization methods
- **Clean API**: Intuitive interface for model building and training

## Performance Highlights

SmartDNN leverages templated C++ to deliver exceptional performance gains:

### Linear Regression Model (1000 samples, 1000 epochs)
- **Non-templated runtime**: ~17680ms
- **Optimized templated runtime**: ~8325ms
- **Performance gain**: ~53% improvement

### MNIST Classification (1000 samples, batch size: 64, 1000 epochs)
- **Non-templated runtime**: ~83 minutes per epoch
- **Optimized templated runtime**: ~10969ms per epoch
- **Performance gain**: ~99.8% improvement

## Getting Started

Creating your first neural network with SmartDNN is straightforward:

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
- **Activation Layers**: ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU
- **Regularization Layers**: Dropout, Batch Normalization
- **Pooling Layers**: Max Pooling 2D
- **Utility Layers**: Flatten

### Optimizers
- **Adam**: Adaptive Moment Estimation optimizer with configurable parameters

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
- **Parallel Directives**: Optimized parallelization for computationally expensive operations

## Installation

### Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/A-Georgiou/SmartDNN.git
   ```

2. Build the library:
   ```bash
   cmake .
   ```

3. Create a `src/main.cpp` file for your neural network code

4. Build your project:
   ```bash
   make
   ```

5. Run the program:
   ```bash
   ./SmartDNN
   ```

### Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/A-Georgiou/SmartDNN.git
   ```

2. Create a `src/main.cpp` file (or copy from the `Examples/` folder)

3. Build the Docker image:
   ```bash
   docker build -f .docker/Dockerfile -t smartdnn-app .
   ```

4. Run the project:
   ```bash
   docker run --rm -it smartdnn-app
   ```

## Example Models

- [Simple Linear Regression model](https://github.com/A-Georgiou/SmartDNN/blob/main/Examples/SimpleLinearRegressionModel.cpp)
- [CNN for MNIST Classification](https://github.com/A-Georgiou/SmartDNN/blob/main/Examples/MNistModel.cpp)

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

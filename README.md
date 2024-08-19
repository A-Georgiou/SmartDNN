
# SmartDNN - A High-Performance C++ Deep Learning Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

SmartDNN is a modern C++ deep learning library designed to offer a flexible and efficient framework for developing and training deep neural networks. With a focus on providing a high-level API, SmartDNN simplifies the process of building and training various neural network architectures while maintaining the performance advantages of C++.

## Getting Started

Creating your first neural network with SmartDNN is straightforward. Define your model's architecture using a variety of available layers, compile it with your chosen loss function and optimizer, and then train it using your dataset.

```cpp
int epochs = 100;
float learningRate = 0.001f;

SmartDNN model;

model.addLayer(new FullyConnectedLayer(10, 100));
model.addLayer(new ActivationLayer(new ReLU()));
model.addLayer(new FullyConnectedLayer(100, 100));
model.addLayer(new ActivationLayer(new Sigmoid()));
model.addLayer(new FullyConnectedLayer(100, 10));
model.addLayer(new ActivationLayer(new Softmax()));

model.compile(new MSELoss(), new AdamOptimizer());
model.train(dataset.first, dataset.second, epochs, learningRate);
```

## Key Features

-   **Custom Tensor Library**: A robust and feature-rich tensor library with comprehensive tensor operations.
-   **Testing Environment**: A built-in testing environment that facilitates clean and efficient development.
-   **Layers**: Includes essential layers such as Fully Connected and Activation layers.
-   **Optimizers**: Currently supports the Adam optimizer.
-   **Loss Functions**: Implements Mean Squared Error (MSE) for regression tasks.
-   **Activation Functions**: Includes popular activation functions like Softmax, Sigmoid, Tanh, ReLU, and Leaky ReLU.

## Roadmap

-   **Extended Layer Support**: Upcoming support for additional layers, including Convolutional, Recurrent, and more.
-   **Advanced Network Architectures**: Flexible and customizable network architectures with user-friendly APIs.
-   **GPU Acceleration**: Future integration of CUDA for efficient GPU-based training and inference.
-   **Comprehensive Documentation**: Detailed documentation and examples to accelerate your development process.


## Installation

To install the library, follow these steps:

1. Clone the repository: `git clone https://github.com/A-Georgiou/SmartDNN.git`

2. Build the library by running the command `CMake .` which will install required dependencies.

3. Create a `src/main.cpp` file to create your neural networks.

3. Build your project from the Makefile with the command `make`

4. Run the program with command `./SmartDNN`

## Contributing

Contributions are welcome! If you would like to contribute to the project, please reach out to me via my contact information below.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or inquiries, please contact [AndrewGeorgiou98@outlook.com](mailto:andrewgeorgiou98@outlook.com).
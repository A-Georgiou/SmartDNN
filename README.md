# SmartDNN

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/language-C%2B%2B17-orange.svg)](https://isocpp.org/)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/a-georgiou/SmartDNN)
[![Tests](https://img.shields.io/badge/tests-105%20passing-brightgreen.svg)](tests/)

A high-performance, header-only C++ deep learning library designed for flexibility, efficiency, and ease of use.

## Overview

SmartDNN is a modern C++17 deep learning framework that offers a clean, intuitive API for building and training neural networks while maintaining C++'s performance advantages. Built with a template-based architecture, it provides zero-overhead abstractions and compile-time optimizations. The library focuses on providing a high-level interface that simplifies neural network development without sacrificing computational efficiency.

### Key Features

- **Header-Only Library**: Simple integration - just include and use
- **Template-Based Design**: Type-safe with zero-overhead abstractions
- **Flexible Architecture**: Easily build and customize neural network architectures
- **High Performance**: Optimized C++ implementation with significant runtime improvements (53-99.8%)
- **Comprehensive Layer Support**: Full suite of essential neural network layers
- **Advanced Tensor Operations**: Efficient tensor computations with broadcasting and slicing
- **Multiple Optimizers**: Adam, SGD, RMSProp with configurable parameters
- **Regularization Support**: Dropout, Batch Normalization, and L1/L2 regularization
- **Customizable Training**: Multiple loss functions and optimization methods
- **Clean API**: Intuitive interface for model building and training
- **Extensively Tested**: 105 unit tests covering all major components

## Table of Contents

- [Performance Highlights](#performance-highlights)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Core Components](#core-components)
- [API Reference](#api-reference)
- [Advanced Examples](#advanced-examples)
- [Architecture](#architecture)
- [Testing](#testing)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)

## Performance Highlights

SmartDNN leverages templated C++ and advanced optimization techniques to deliver exceptional performance gains:

### Linear Regression Model (1000 samples, 1000 epochs)
- **Non-templated runtime**: ~17680ms
- **Optimized templated runtime**: ~8325ms
- **Performance gain**: ~53% improvement

### MNIST Classification (1000 samples, batch size: 64, 1000 epochs)
- **Non-templated runtime**: ~83 minutes per epoch
- **Optimized templated runtime**: ~10969ms per epoch
- **Performance gain**: ~99.8% improvement

These improvements are achieved through:
- Template metaprogramming for compile-time optimizations
- Iterator-based tensor transforms enabling compiler vectorization
- Efficient memory management with SliceView and BroadcastView
- Parallel directives for computationally intensive operations

## Quick Start

Creating your first neural network with SmartDNN is straightforward:

```cpp
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Activations/Softmax.hpp"
#include "smart_dnn/Loss/MSELoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"

using namespace smart_dnn;

// Initialize the model
SmartDNN<float> model;

// Define architecture
model.addLayer(FullyConnectedLayer(10, 100));        // Input -> Hidden
model.addLayer(ActivationLayer(ReLU()));             // ReLU activation
model.addLayer(FullyConnectedLayer(100, 100));       // Hidden -> Hidden
model.addLayer(ActivationLayer(ReLU()));             // ReLU activation
model.addLayer(FullyConnectedLayer(100, 10));        // Hidden -> Output
model.addLayer(ActivationLayer(Softmax()));          // Softmax for classification

// Configure optimizer
AdamOptions adamOptions;
adamOptions.learningRate = 0.001f;

// Compile and train
model.compile(MSELoss(), AdamOptimizer(adamOptions));
model.train(inputs, targets, epochs);

// Make predictions
model.evalMode();
Tensor<float> prediction = model.predict(input);
```

## API Reference

### SmartDNN Class

The main model class for building and training neural networks.

```cpp
template <typename T = float>
class SmartDNN {
public:
    // Add a layer to the model
    template<typename LayerType>
    void addLayer(LayerType&& layer);
    
    // Compile model with loss function and optimizer
    template<typename LossType, typename OptimizerType>
    void compile(LossType&& loss, OptimizerType&& optimizer);
    
    // Train the model
    void train(const std::vector<Tensor<T>>& inputs, 
               const std::vector<Tensor<T>>& targets, 
               int epochs);
    
    // Make predictions
    Tensor<T> predict(const Tensor<T>& input);
    std::vector<Tensor<T>> predict(const std::vector<Tensor<T>>& inputs);
    
    // Set model mode
    void trainingMode();  // Enable dropout, batch norm training mode
    void evalMode();      // Disable dropout, use batch norm statistics
    
    // Model persistence
    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);
    
    // Access layers
    Layer<T>* getLayer(size_t index) const;
};
```

### Training Workflow

```cpp
// 1. Create model
SmartDNN<float> model;

// 2. Add layers
model.addLayer(FullyConnectedLayer(784, 128));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(DropoutLayer(0.2f));
model.addLayer(FullyConnectedLayer(128, 10));
model.addLayer(ActivationLayer(Softmax()));

// 3. Compile with loss and optimizer
model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer(adamOptions));

// 4. Train
model.train(trainInputs, trainTargets, epochs);

// 5. Switch to evaluation mode
model.evalMode();

// 6. Make predictions
auto predictions = model.predict(testInputs);
```

## Advanced Examples

### Linear Regression Example

```cpp
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Loss/MSELoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "smart_dnn/Datasets/SampleGenerator.hpp"

using namespace smart_dnn;

int main() {
    constexpr int BATCH_SIZE = 100;
    constexpr int EPOCHS = 100;
    constexpr float LEARNING_RATE = 0.01f;

    // Generate linear dataset: y = 2x + 3 + noise
    auto [inputs, targets] = generateLinearDataset(BATCH_SIZE);

    // Build model
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer(1, 10));
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(FullyConnectedLayer(10, 1));

    // Compile and train
    AdamOptions adamOptions;
    adamOptions.learningRate = LEARNING_RATE;
    model.compile(MSELoss(), AdamOptimizer(adamOptions));
    model.train(inputs, targets, EPOCHS);

    // Predict
    model.evalMode();
    Tensor input(Shape{1}, 10.0f);
    Tensor prediction = model.predict(input);
    std::cout << "Input: 10.0 | Prediction: " << prediction.toDetailedString() << std::endl;
    
    return 0;
}
```

### MNIST CNN Model

```cpp
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/Conv2DLayer.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Layers/FlattenLayer.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Activations/Softmax.hpp"
#include "smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "smart_dnn/Regularisation/DropoutLayer.hpp"
#include "smart_dnn/Regularisation/MaxPooling2DLayer.hpp"
#include "smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "smart_dnn/Datasets/MNistLoader.hpp"

using namespace smart_dnn;

int main() {
    constexpr int EPOCHS = 10;
    constexpr int BATCH_SIZE = 8;
    constexpr int SAMPLE_COUNT = 1000;
    constexpr float LEARNING_RATE = 0.001f;

    // Initialize the SmartDNN MNIST model
    SmartDNN<float> model;

    // Convolutional layers
    model.addLayer(Conv2DLayer(1, 32, 3));           // Conv2D layer
    model.addLayer(BatchNormalizationLayer(32));     // Batch normalization
    model.addLayer(ActivationLayer(ReLU()));         // ReLU activation
    model.addLayer(MaxPooling2DLayer(2, 2));         // MaxPooling
    model.addLayer(DropoutLayer(0.25f));             // Dropout for regularization

    // Fully connected layers
    model.addLayer(FlattenLayer());                  // Flatten layer
    model.addLayer(FullyConnectedLayer(5408, 128));  // FC layer
    model.addLayer(BatchNormalizationLayer(128));    // Batch normalization
    model.addLayer(ActivationLayer(ReLU()));         // ReLU activation
    model.addLayer(DropoutLayer(0.25f));             // Dropout

    // Output layer
    model.addLayer(FullyConnectedLayer(128, 10));    // Output layer
    model.addLayer(ActivationLayer(Softmax()));      // Softmax activation

    // Configure optimizer
    AdamOptions adamOptions;
    adamOptions.learningRate = LEARNING_RATE;
    adamOptions.beta1 = 0.9f;
    adamOptions.beta2 = 0.999f;
    adamOptions.epsilon = 1e-8f;
    
    model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer(adamOptions));

    // Load MNIST dataset (download from http://yann.lecun.com/exdb/mnist/)
    std::string imagesPath = ".datasets/train-images-idx3-ubyte";
    std::string labelsPath = ".datasets/train-labels-idx1-ubyte";
    MNISTLoader dataLoader(imagesPath, labelsPath, BATCH_SIZE, SAMPLE_COUNT);
    auto [inputs, targets] = dataLoader.loadData();

    // Train the model
    model.train(inputs, targets, EPOCHS);

    // Evaluate
    model.evalMode();
    MNISTLoader testLoader(imagesPath, labelsPath, BATCH_SIZE);
    auto [testInputs, testTargets] = testLoader.loadData();
    
    for (size_t i = testInputs.size()-5; i < testInputs.size(); i++) {
        Tensor prediction = model.predict(testInputs[i]);
        std::cout << "Prediction: " << prediction.toDataString() << std::endl;
        std::cout << "Actual: " << testTargets[i].toDataString() << std::endl;
    }

    return 0;
}
```

### Custom Training Loop

For more control over training, you can implement a custom training loop:

```cpp
SmartDNN<float> model;
// ... add layers and compile ...

model.trainingMode();
for (int epoch = 0; epoch < epochs; ++epoch) {
    float totalLoss = 0.0f;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Forward pass
        Tensor<float> prediction = model.predict(inputs[i]);
        
        // Compute loss (manual)
        // ... your loss computation ...
        
        // Backward pass (if you need custom gradients)
        // model.backward(gradients);
        // model.updateWeights();
        
        totalLoss += loss;
    }
    
    std::cout << "Epoch " << epoch << " - Loss: " 
              << totalLoss / inputs.size() << std::endl;
}
model.evalMode();
```

## Core Components

### Tensor Operations

The `Tensor<T>` class is the fundamental data structure in SmartDNN, supporting:

```cpp
using namespace smart_dnn;

// Creating tensors
Tensor<float> t1(Shape({3, 4}));              // 3x4 tensor of zeros
Tensor<float> t2(Shape({3, 4}), 1.0f);        // 3x4 tensor filled with 1.0
Tensor<float> t3 = Tensor<float>::ones({3, 4}); // Static factory method
Tensor<float> t4 = Tensor<float>::rand({3, 4}); // Random initialization

// Tensor operations
Tensor<float> sum = t1 + t2;                  // Element-wise addition
Tensor<float> product = t1 * t2;              // Element-wise multiplication
Tensor<float> scaled = t1 * 2.0f;             // Scalar multiplication

// Reshaping
t1.reshape({4, 3});                           // Reshape to 4x3

// Advanced operations
Tensor<float> matmul_result = AdvancedTensorOperations<float>::matmul(t1, t2);
Tensor<float> transposed = AdvancedTensorOperations<float>::transpose(t1);
```

### Available Layers

#### Fully Connected Layer
Dense neural network layer with learnable weights and biases.

```cpp
FullyConnectedLayer<float>(inputSize, outputSize)
```

**Input**: 1D or 2D tensor `(input_size)` or `(batch_size, input_size)`  
**Output**: 1D or 2D tensor `(output_size)` or `(batch_size, output_size)`

#### Convolutional 2D Layer
2D convolution layer for image processing tasks.

```cpp
Conv2DLayer<float>(inputChannels, outputChannels, kernelSize)
Conv2DLayer<float>(inputChannels, outputChannels, kernelHeight, kernelWidth, stride, padding, dilation)
```

**Input**: 4D tensor `(batch_size, input_channels, height, width)`  
**Output**: 4D tensor `(batch_size, output_channels, out_height, out_width)`

#### Activation Layer
Wraps activation functions for use in the model.

```cpp
ActivationLayer(ReLU())
ActivationLayer(Sigmoid())
ActivationLayer(Tanh())
ActivationLayer(Softmax())
ActivationLayer(LeakyReLU(alpha))
ActivationLayer(Swish())
ActivationLayer(Mish())
```

#### Flatten Layer
Flattens multi-dimensional input to 1D.

```cpp
FlattenLayer<float>()
```

**Input**: nD tensor `(batch_size, dim1, dim2, ...)`  
**Output**: 2D tensor `(batch_size, dim1*dim2*...)`

#### Regularization Layers

**Dropout Layer**: Randomly zeros elements during training
```cpp
DropoutLayer<float>(dropoutRate)  // e.g., 0.25f for 25% dropout
```

**Batch Normalization Layer**: Normalizes layer inputs
```cpp
BatchNormalizationLayer<float>(numFeatures)
```

**Max Pooling 2D Layer**: Downsamples spatial dimensions
```cpp
MaxPooling2DLayer<float>(poolHeight, poolWidth)
MaxPooling2DLayer<float>(poolSize)  // Square pooling
```

### Activation Functions

All activation functions support both forward and backward passes for gradient computation.

| Activation | Usage | Description |
|------------|-------|-------------|
| **ReLU** | `ReLU()` | Rectified Linear Unit: `max(0, x)` |
| **Leaky ReLU** | `LeakyReLU(alpha)` | Leaky ReLU with configurable negative slope |
| **Sigmoid** | `Sigmoid()` | Logistic sigmoid: `1 / (1 + e^-x)` |
| **Tanh** | `Tanh()` | Hyperbolic tangent: `(e^x - e^-x) / (e^x + e^-x)` |
| **Softmax** | `Softmax()` | Normalized exponential for multi-class classification |
| **Swish** | `Swish()` | Self-gated activation: `x * sigmoid(x)` |
| **Mish** | `Mish()` | Smooth activation: `x * tanh(softplus(x))` |

### Optimizers

#### Adam Optimizer
Adaptive Moment Estimation optimizer with momentum and adaptive learning rates.

```cpp
AdamOptions adamOptions;
adamOptions.learningRate = 0.001f;
adamOptions.beta1 = 0.9f;          // First moment decay
adamOptions.beta2 = 0.999f;        // Second moment decay
adamOptions.epsilon = 1e-8f;       // Numerical stability
adamOptions.l1Strength = 0.0f;     // L1 regularization
adamOptions.l2Strength = 0.0f;     // L2 regularization
adamOptions.decay = 0.0f;          // Learning rate decay

AdamOptimizer optimizer(adamOptions);
```

#### SGD Optimizer
Stochastic Gradient Descent with optional momentum and Nesterov acceleration.

```cpp
SGDOptions sgdOptions;
sgdOptions.learningRate = 0.01f;
sgdOptions.momentum = 0.9f;        // Momentum coefficient
sgdOptions.nesterov = true;        // Use Nesterov momentum
sgdOptions.dampening = 0.0f;       // Dampening for momentum
sgdOptions.l1Strength = 0.0f;      // L1 regularization
sgdOptions.l2Strength = 0.0001f;   // L2 regularization (weight decay)
sgdOptions.decay = 0.0f;           // Learning rate decay

SGDOptimizer optimizer(sgdOptions);
```

#### RMSProp Optimizer
Root Mean Square Propagation optimizer.

```cpp
RMSPropOptions rmspropOptions;
rmspropOptions.learningRate = 0.001f;
rmspropOptions.alpha = 0.99f;      // Smoothing constant
rmspropOptions.epsilon = 1e-8f;    // Numerical stability
rmspropOptions.momentum = 0.0f;    // Momentum factor
rmspropOptions.centered = false;   // Use centered RMSProp
rmspropOptions.l1Strength = 0.0f;  // L1 regularization
rmspropOptions.l2Strength = 0.0f;  // L2 regularization
rmspropOptions.decay = 0.0f;       // Learning rate decay

RMSPropOptimizer optimizer(rmspropOptions);
```

### Loss Functions
- **Mean Squared Error (MSE)**: For regression tasks
  ```cpp
  MSELoss()
  ```
- **Categorical Cross Entropy**: For multi-class classification
  ```cpp
  CategoricalCrossEntropyLoss()
  ```

## Architecture

### Design Principles

SmartDNN follows several key design principles:

1. **Header-Only Design**: Entire library is header-only for easy integration and maximum compile-time optimization opportunities.

2. **Template Metaprogramming**: Extensive use of C++ templates allows for:
   - Type safety without runtime overhead
   - Compile-time optimizations
   - Zero-cost abstractions

3. **Single Responsibility Principle**: Each class has a well-defined, single purpose:
   - `Tensor<T>`: Data storage and basic operations
   - `Layer<T>`: Neural network layer interface
   - `Optimizer<T>`: Parameter update strategies
   - `Loss<T>`: Loss computation and gradient calculation

4. **Efficient Memory Management**: 
   - **SliceView**: Access tensor slices without copying data
   - **BroadcastView**: Efficient broadcasting for element-wise operations
   - Move semantics throughout for optimal performance

### Project Structure

```
SmartDNN/
├── smart_dnn/
│   ├── Tensor/              # Tensor operations and data structures
│   │   ├── Tensor.hpp       # Core tensor class
│   │   ├── TensorData.hpp   # Data storage abstraction
│   │   ├── TensorOperations.hpp
│   │   ├── AdvancedTensorOperations.hpp
│   │   ├── SliceView.hpp    # Zero-copy tensor slicing
│   │   └── BroadcastView.hpp # Broadcasting support
│   ├── Layers/              # Neural network layers
│   │   ├── Layer.hpp        # Base layer interface
│   │   ├── FullyConnectedLayer.hpp
│   │   ├── Conv2DLayer.hpp
│   │   ├── ActivationLayer.hpp
│   │   └── FlattenLayer.hpp
│   ├── Activations/         # Activation functions
│   │   ├── ReLU.hpp
│   │   ├── Sigmoid.hpp
│   │   ├── Tanh.hpp
│   │   ├── Softmax.hpp
│   │   ├── LeakyReLU.hpp
│   │   ├── Swish.hpp
│   │   └── Mish.hpp
│   ├── Optimizers/          # Optimization algorithms
│   │   ├── AdamOptimizer.hpp
│   │   ├── SGDOptimizer.hpp
│   │   └── RMSPropOptimizer.hpp
│   ├── Loss/                # Loss functions
│   │   ├── MSELoss.hpp
│   │   └── CategoricalCrossEntropyLoss.hpp
│   ├── Regularisation/      # Regularization techniques
│   │   ├── DropoutLayer.hpp
│   │   ├── BatchNormalizationLayer.hpp
│   │   └── MaxPooling2DLayer.hpp
│   ├── Datasets/            # Dataset utilities
│   │   ├── MNistLoader.hpp
│   │   └── SampleGenerator.hpp
│   ├── Shape/               # Shape handling
│   │   └── Shape.hpp
│   └── SmartDNN.hpp         # Main model class
├── examples/                # Example implementations
├── tests/                   # Unit tests (105 tests)
└── CMakeLists.txt          # Build configuration
```

### Data Flow

```
Input Tensor
     ↓
[Layer 1: Forward Pass]
     ↓
[Layer 2: Forward Pass]
     ↓
    ...
     ↓
[Layer N: Forward Pass]
     ↓
Output Tensor
     ↓
[Loss Computation]
     ↓
Loss Gradient
     ↓
[Layer N: Backward Pass]
     ↓
[Layer N-1: Backward Pass]
     ↓
    ...
     ↓
[Layer 1: Backward Pass]
     ↓
[Optimizer: Update Weights]
```

## Installation

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10 or higher
- (Optional) Docker for containerized builds

### Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/A-Georgiou/SmartDNN.git
   cd SmartDNN
   ```

2. Create your source file:
   ```bash
   mkdir -p src
   # Create src/main.cpp with your neural network code
   ```

3. Build the library:
   ```bash
   cmake .
   make
   ```

4. Run your program:
   ```bash
   ./SmartDNN
   ```

### Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/A-Georgiou/SmartDNN.git
   cd SmartDNN
   ```

2. Create a `src/main.cpp` file (or copy from the `examples/` folder)

3. Build the Docker image:
   ```bash
   docker build -f .docker/Dockerfile -t smartdnn-app .
   ```

4. Run the project:
   ```bash
   docker run --rm -it smartdnn-app
   ```

### Integration into Existing Project

Since SmartDNN is header-only, you can integrate it into your existing CMake project:

```cmake
# In your CMakeLists.txt
include_directories(${PROJECT_SOURCE_DIR}/path/to/SmartDNN)

# Then in your source files
#include "smart_dnn/SmartDNN.hpp"
```

### Building Tests

```bash
cd tests
mkdir build && cd build
cmake ..
make
./RunTests
```

## Testing

SmartDNN includes a comprehensive test suite with **105 unit tests** covering all major components.

### Running Tests

```bash
cd tests
mkdir build && cd build
cmake ..
make
./RunTests
```

### Test Coverage

The test suite covers:

- **Tensor Operations** (25 tests)
  - Basic arithmetic operations
  - Advanced tensor operations (matmul, transpose, etc.)
  - Shape manipulation and broadcasting
  - Copy/move semantics

- **Activation Functions** (14 tests)
  - Forward and backward passes for all activation functions
  - Gradient verification

- **Layers** (17 tests)
  - Fully connected layer forward/backward passes
  - Conv2D layer operations
  - Weight initialization and updates

- **Optimizers** (12 tests)
  - Adam optimizer
  - SGD optimizer (with/without momentum, Nesterov)
  - RMSProp optimizer (basic, centered, with momentum)

- **Utility Functions** (37 tests)
  - Shape operations
  - Tensor slicing and broadcasting
  - Data transformations

### Test Results

```
[==========] Running 105 tests from 18 test suites.
...
[==========] 105 tests from 18 test suites ran. (56 ms total)
[  PASSED  ] 105 tests.
```

### Writing Tests

Tests use Google Test framework. Example test:

```cpp
TEST(TensorTest, BasicAddition) {
    Tensor<float> t1(Shape({2, 2}), 1.0f);
    Tensor<float> t2(Shape({2, 2}), 2.0f);
    Tensor<float> result = t1 + t2;
    
    EXPECT_EQ(result[0], 3.0f);
    EXPECT_EQ(result[1], 3.0f);
    EXPECT_EQ(result[2], 3.0f);
    EXPECT_EQ(result[3], 3.0f);
}
```

## Performance Optimization

SmartDNN achieves high performance through several key optimization techniques:

### 1. Template Metaprogramming
- **Zero-overhead abstractions**: Templates allow compile-time polymorphism without virtual function overhead
- **Type-specific optimizations**: Compiler can generate optimized code for each type
- **Inline expansion**: Header-only design enables aggressive inlining

### 2. Memory Optimization

**Slice View**: Access tensor slices without copying data
```cpp
// Zero-copy slicing
Tensor<float> slice = tensor.slice(dim, index);
```

**Broadcast View**: Efficient broadcasting for element-wise operations
```cpp
// Efficient broadcasting without data duplication
Tensor<float> result = tensor1 + broadcastedTensor2;
```

### 3. Computational Optimizations

- **Iterator-based transforms**: Enable compiler auto-vectorization (SIMD)
- **Cache-friendly memory access**: Contiguous memory layout
- **Parallel directives**: Multi-threading for large tensor operations
- **Move semantics**: Eliminate unnecessary copies throughout the codebase

### 4. Build Optimizations

The CMake configuration includes compiler-specific optimizations:

```cmake
# For GCC/Clang
-O3 -march=native

# For MSVC
/O2
```

### Performance Tips

1. **Use Release builds**: Debug builds are significantly slower
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release .
   ```

2. **Enable native architecture optimizations**: Use `-march=native` flag

3. **Batch processing**: Process multiple samples together when possible

4. **Appropriate data types**: Use `float` instead of `double` when precision allows

5. **Pre-allocate tensors**: Reuse tensor objects to avoid allocations in hot loops

## Dataset Utilities

SmartDNN includes utilities for loading and generating datasets.

### MNIST Loader

Load the MNIST handwritten digit dataset:

```cpp
#include "smart_dnn/Datasets/MNistLoader.hpp"

// Download MNIST from http://yann.lecun.com/exdb/mnist/
std::string imagesPath = ".datasets/train-images-idx3-ubyte";
std::string labelsPath = ".datasets/train-labels-idx1-ubyte";

// Load with batch size and sample count
MNISTLoader loader(imagesPath, labelsPath, batchSize, sampleCount);
auto [inputs, targets] = loader.loadData();

// Convert image to ASCII art for visualization
std::cout << loader.toAsciiArt(inputs[0]) << std::endl;
```

### Sample Generator

Generate synthetic datasets for testing:

```cpp
#include "smart_dnn/Datasets/SampleGenerator.hpp"

// Generate linear regression dataset: y = 2x + 3 + noise
auto [inputs, targets] = generateLinearDataset(numSamples);
```

## FAQ

### General Questions

**Q: Is SmartDNN production-ready?**  
A: SmartDNN is an educational/research framework. While it's well-tested and functional, it may not have all features needed for production deployments.

**Q: Does SmartDNN support GPU acceleration?**  
A: Not currently. GPU support (CUDA) is on the roadmap for future development.

**Q: What platforms are supported?**  
A: SmartDNN works on Linux, macOS, and Windows with any C++17 compatible compiler.

**Q: Can I use SmartDNN in my project?**  
A: Yes! SmartDNN is licensed under the MIT License, allowing commercial and non-commercial use.

### Technical Questions

**Q: How do I debug training issues?**  
A: Build in Debug mode to enable logging:
```cmake
cmake -DCMAKE_BUILD_TYPE=Debug .
```

**Q: My model isn't converging. What should I check?**  
A:
1. Learning rate (try 0.001, 0.01, 0.1)
2. Weight initialization
3. Gradient clipping if gradients explode
4. Network architecture (too deep/shallow)
5. Data normalization

**Q: How do I save and load models?**  
A: Use the model persistence methods:
```cpp
// Save model
model.saveModel("model.bin");

// Load model
model.loadModel("model.bin");
```

**Q: Can I use custom activation functions?**  
A: Yes! Implement the `Activation<T>` interface:
```cpp
template <typename T = float>
class MyActivation : public Activation<T> {
    Tensor<T> forward(const Tensor<T>& input) override { /* ... */ }
    Tensor<T> backward(const Tensor<T>& gradOutput) override { /* ... */ }
};
```

**Q: How do I handle different input shapes?**  
A: Most layers handle reshaping automatically. For specific needs:
```cpp
tensor.reshape({newDim1, newDim2});
```

## Example Models

Complete working examples are available in the `examples/` directory:

- **[Simple Linear Regression](examples/SimpleLinearRegressionModel.cpp)**: Basic regression task
- **[MNIST CNN Classifier](examples/MNistModel.cpp)**: Image classification with convolutional networks

## Troubleshooting

### Common Build Issues

**Issue**: `fatal error: smart_dnn/SmartDNN.hpp: No such file or directory`  
**Solution**: Ensure the include path is correctly set in CMakeLists.txt:
```cmake
include_directories(${PROJECT_SOURCE_DIR})
```

**Issue**: Compiler errors about C++17 features  
**Solution**: Verify compiler version and C++ standard:
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

**Issue**: Tests fail to build  
**Solution**: Make sure Google Test is downloaded:
```bash
cd tests/build
rm -rf *
cmake ..
make
```

### Runtime Issues

**Issue**: Segmentation fault during training  
**Solution**: Check tensor dimensions match between layers:
```cpp
// Output of previous layer must match input of next layer
model.addLayer(FullyConnectedLayer(inputDim, hiddenDim));
model.addLayer(FullyConnectedLayer(hiddenDim, outputDim)); // hiddenDim must match
```

**Issue**: NaN values in loss  
**Solution**: 
- Reduce learning rate
- Check for division by zero
- Add gradient clipping
- Verify data normalization

**Issue**: Very slow training  
**Solution**:
- Build in Release mode
- Enable compiler optimizations
- Use smaller batch sizes
- Consider network architecture complexity

## Roadmap

### Planned Features

#### Short Term
- [ ] Additional activation functions (ELU, GELU, PReLU)
- [ ] More loss functions (Binary Cross Entropy, Huber Loss)
- [ ] Learning rate schedulers (Step Decay, Exponential Decay, Cosine Annealing)
- [ ] Model checkpointing during training
- [ ] Gradient clipping utilities

#### Medium Term
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Additional convolutional layer types (DepthwiseConv2D, SeparableConv2D)
- [ ] Advanced pooling (Average Pooling, Global Pooling)
- [ ] Data augmentation utilities
- [ ] Pretrained model zoo
- [ ] Improved serialization format
- [ ] Training callbacks system

#### Long Term
- [ ] GPU acceleration (CUDA support)
- [ ] Distributed training capabilities
- [ ] Automatic differentiation engine improvements
- [ ] Graph-based computation model
- [ ] ONNX export support
- [ ] Quantization and model compression
- [ ] Mobile deployment support

## Contributing

Contributions are welcome! We appreciate any help in improving SmartDNN.

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/A-Georgiou/SmartDNN.git
   cd SmartDNN
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow the existing code style
   - Add tests for new features
   - Update documentation as needed

4. **Test your changes**
   ```bash
   cd tests/build
   cmake ..
   make
   ./RunTests
   ```

5. **Submit a pull request**
   - Describe your changes clearly
   - Reference any related issues
   - Ensure all tests pass

### Contribution Guidelines

- **Code Style**: Follow modern C++ best practices
- **Documentation**: Document all public APIs
- **Testing**: Add unit tests for new features
- **Commits**: Write clear, descriptive commit messages
- **Performance**: Avoid performance regressions

### Areas for Contribution

- **Bug Fixes**: Find and fix bugs
- **New Features**: Implement items from the roadmap
- **Documentation**: Improve documentation and examples
- **Testing**: Increase test coverage
- **Performance**: Optimize existing implementations
- **Examples**: Add more example models and use cases

### Getting Help

For questions or discussions:
- Open an issue on GitHub
- Contact: [AndrewGeorgiou98@outlook.com](mailto:andrewgeorgiou98@outlook.com)

## Citation

If you use SmartDNN in your research or project, please cite:

```bibtex
@software{smartdnn2024,
  author = {Georgiou, Andrew},
  title = {SmartDNN: A High-Performance C++ Deep Learning Library},
  year = {2024},
  url = {https://github.com/A-Georgiou/SmartDNN}
}
```

## Acknowledgments

SmartDNN is built using modern C++ best practices and draws inspiration from popular deep learning frameworks while maintaining a focus on performance and simplicity.

## License

This project is licensed under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2024 Andrew Georgiou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

**Author**: Andrew Georgiou  
**Email**: [AndrewGeorgiou98@outlook.com](mailto:andrewgeorgiou98@outlook.com)  
**GitHub**: [A-Georgiou/SmartDNN](https://github.com/A-Georgiou/SmartDNN)

---

**SmartDNN** - High-performance deep learning in modern C++

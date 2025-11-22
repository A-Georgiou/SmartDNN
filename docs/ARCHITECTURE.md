# SmartDNN Architecture Guide

This document provides an in-depth overview of SmartDNN's architecture, design principles, and component interactions.

## Table of Contents

- [Overview](#overview)
- [Design Philosophy](#design-philosophy)
- [Core Architecture](#core-architecture)
- [Component Hierarchy](#component-hierarchy)
- [Data Flow](#data-flow)
- [Memory Management](#memory-management)
- [Performance Optimizations](#performance-optimizations)
- [Extension Points](#extension-points)

---

## Overview

SmartDNN is a header-only C++ deep learning library built on modern C++17 features. The architecture emphasizes:

- **Type Safety**: Template-based design for compile-time type checking
- **Performance**: Zero-cost abstractions and compiler optimizations
- **Modularity**: Clean separation of concerns with well-defined interfaces
- **Extensibility**: Easy to add new layers, optimizers, and loss functions

The library follows a layered architecture where each component has a specific responsibility and well-defined interfaces.

---

## Design Philosophy

### 1. Template-Based Design

SmartDNN heavily uses C++ templates to achieve:

- **Type Flexibility**: Support for different numeric types (float, double)
- **Zero-Cost Abstractions**: Template instantiation at compile time
- **Type Safety**: Compile-time type checking
- **Performance**: Inline expansion and compiler optimizations

```cpp
template <typename T=float>
class SmartDNN { ... }

template <typename T=float>
class Tensor { ... }

template <typename T=float>
class Layer { ... }
```

### 2. RAII (Resource Acquisition Is Initialization)

All resources are managed through constructors and destructors:

- Automatic memory management
- No manual memory deallocation needed
- Exception-safe resource handling

### 3. Single Responsibility Principle

Each class has a single, well-defined purpose:

- **Tensor**: Multi-dimensional array storage and operations
- **Layer**: Neural network layer logic (forward/backward)
- **Optimizer**: Weight update strategies
- **Loss**: Loss computation and gradient calculation

### 4. Interface Segregation

Abstract base classes define minimal interfaces:

```cpp
template <typename T>
class Layer {
    virtual Tensor<T> forward(const Tensor<T>& input) = 0;
    virtual Tensor<T> backward(const Tensor<T>& gradOutput) = 0;
    // Optional methods have default implementations
};
```

---

## Core Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        SmartDNN Model                       │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              Layer Stack (std::vector)                │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │ │
│  │  │  Layer 1 │→ │  Layer 2 │→ │  Layer N │            │ │
│  │  └──────────┘  └──────────┘  └──────────┘            │ │
│  └───────────────────────────────────────────────────────┘ │
│  ┌──────────────┐           ┌────────────────────────┐    │
│  │ Loss Function│           │     Optimizer          │    │
│  └──────────────┘           └────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Tensor Engine   │
                    │  ┌────────────┐  │
                    │  │   Data     │  │
                    │  │   Shape    │  │
                    │  │ Operations │  │
                    │  └────────────┘  │
                    └──────────────────┘
```

### Component Layers

1. **Application Layer**: User-facing API (`SmartDNN` class)
2. **Computation Layer**: Layers, activations, loss functions
3. **Optimization Layer**: Optimizers and weight updates
4. **Tensor Layer**: Multi-dimensional arrays and operations
5. **Utility Layer**: Shape, random generators, data loaders

---

## Component Hierarchy

### 1. Tensor Subsystem

The foundation of the library, providing multi-dimensional array support.

```
Tensor<T>
├── TensorData<T, DeviceType>
│   ├── TensorDataCPU (Implementation for CPU)
│   └── TensorDataGPU (Future: GPU implementation)
├── Shape
│   └── ShapeOperations
├── TensorOperations
│   ├── Element-wise operations (+, -, *, /)
│   └── Reductions (sum, mean, max, min)
└── AdvancedTensorOperations
    ├── Matrix multiplication (matmul)
    ├── Convolution operations
    ├── Transpose
    └── Reshape
```

**Key Files:**
- `smart_dnn/Tensor/Tensor.hpp` - Main tensor class
- `smart_dnn/Tensor/TensorData.hpp` - Data storage abstraction
- `smart_dnn/Tensor/TensorOperations.hpp` - Basic operations
- `smart_dnn/Tensor/AdvancedTensorOperations.hpp` - Advanced operations

### 2. Layer Hierarchy

Abstract base class with concrete implementations.

```
Layer<T> (Abstract)
├── FullyConnectedLayer<T>
├── Conv2DLayer<T>
├── ActivationLayer<T>
│   └── Wraps Activation<T> objects
├── FlattenLayer<T>
├── DropoutLayer<T>
├── BatchNormalizationLayer<T>
└── MaxPooling2DLayer<T>
```

**Responsibilities:**
- **Forward Pass**: Transform input to output
- **Backward Pass**: Compute gradients
- **Weight Updates**: Update parameters (if applicable)
- **Mode Management**: Training vs. evaluation mode

**Key Files:**
- `smart_dnn/Layer.hpp` - Base layer interface
- `smart_dnn/Layers/` - Layer implementations

### 3. Activation Functions

```
Activation<T> (Abstract)
├── ReLU<T>
├── Sigmoid<T>
├── Tanh<T>
├── Softmax<T>
├── LeakyReLU<T>
├── Swish<T>
└── Mish<T>
```

**Responsibilities:**
- Apply non-linear transformation
- Compute gradients for backpropagation

**Key Files:**
- `smart_dnn/Activation.hpp` - Base activation interface
- `smart_dnn/Activations/` - Activation implementations

### 4. Optimizer Hierarchy

```
Optimizer<T> (Abstract)
├── AdamOptimizer<T>
├── SGDOptimizer<T>
└── RMSPropOptimizer<T>
```

**Responsibilities:**
- Update weights based on gradients
- Maintain optimizer state (momentum, adaptive rates, etc.)
- Apply regularization (L1, L2)

**Key Files:**
- `smart_dnn/Optimizer.hpp` - Base optimizer interface
- `smart_dnn/Optimizers/` - Optimizer implementations

### 5. Loss Functions

```
Loss<T> (Abstract)
├── MSELoss<T>
└── CategoricalCrossEntropyLoss<T>
```

**Responsibilities:**
- Compute loss value
- Compute loss gradients

**Key Files:**
- `smart_dnn/Loss.hpp` - Base loss interface
- `smart_dnn/Loss/` - Loss implementations

---

## Data Flow

### Training Flow

```
1. Input Data
   │
   ▼
2. Forward Pass (Layer by Layer)
   │ model.train()
   │  ├─→ Layer 1: forward()
   │  ├─→ Layer 2: forward()
   │  └─→ Layer N: forward()
   │
   ▼
3. Loss Computation
   │ lossFunction.compute(prediction, target)
   │
   ▼
4. Gradient Computation
   │ lossFunction.gradient(prediction, target)
   │
   ▼
5. Backward Pass (Reverse Layer Order)
   │  ├─→ Layer N: backward()
   │  ├─→ Layer 2: backward()
   │  └─→ Layer 1: backward()
   │
   ▼
6. Weight Update
   │  ├─→ optimizer.optimize(weights, gradients)
   │  └─→ Update all layer parameters
   │
   └─→ Repeat for next batch/epoch
```

### Inference Flow

```
1. Input Data
   │
   ▼
2. Set Evaluation Mode
   │ model.evalMode()
   │ (Disables dropout, uses batch norm running stats)
   │
   ▼
3. Forward Pass Only
   │ model.predict(input)
   │  ├─→ Layer 1: forward()
   │  ├─→ Layer 2: forward()
   │  └─→ Layer N: forward()
   │
   ▼
4. Return Prediction
```

### Backward Pass Details

Each layer computes gradients with respect to:
1. **Input** (gradInput): Passed to previous layer
2. **Weights** (gradWeights): Used for weight updates
3. **Biases** (gradBiases): Used for bias updates

```cpp
Tensor<T> Layer::backward(const Tensor<T>& gradOutput) {
    // 1. Compute gradient w.r.t. input
    Tensor<T> gradInput = computeInputGradient(gradOutput);
    
    // 2. Compute gradient w.r.t. weights (if applicable)
    if (hasWeights) {
        gradWeights = computeWeightGradient(gradOutput);
        gradBiases = computeBiasGradient(gradOutput);
    }
    
    // 3. Return gradient to pass to previous layer
    return gradInput;
}
```

---

## Memory Management

### Tensor Storage

Tensors use a shared pointer pattern for efficient memory management:

```cpp
template <typename T, typename DeviceType>
class Tensor {
    TensorData<T, DeviceType> data;  // Handles actual data storage
    Shape shape;                      // Dimension information
};
```

**Benefits:**
- Automatic memory cleanup
- Efficient copying (copy-on-write possible)
- Exception-safe

### View Mechanisms

SmartDNN provides views to avoid unnecessary data copying:

1. **SliceView**: Access tensor slices without copying
2. **BroadcastView**: Expand tensors virtually for broadcasting operations

```cpp
// Create a view instead of copying data
auto slice = tensor.slice(0, index);  // Returns a view

// Broadcasting for efficient element-wise operations
auto broadcast = BroadcastView::create(tensor, targetShape);
```

### Layer State Management

Layers store state in member variables:

```cpp
class FullyConnectedLayer : public Layer<T> {
    std::optional<Tensor<T>> weights;
    std::optional<Tensor<T>> biases;
    Tensor<T> input;  // Cached for backward pass
    // ...
};
```

- **Weights/Biases**: Persistent across batches
- **Cached Inputs**: Stored during forward, used in backward
- **Optimizer State**: Maintained by optimizer (momentum, etc.)

---

## Performance Optimizations

### 1. Template Specialization

Different code paths for different types:

```cpp
template <typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

// Specialized version for float (can use SIMD)
template <>
Tensor<float> matmul(const Tensor<float>& a, const Tensor<float>& b);
```

### 2. Parallel Operations

Computationally intensive operations use OpenMP:

```cpp
#pragma omp parallel for
for (size_t i = 0; i < size; ++i) {
    result[i] = op(data[i]);
}
```

### 3. Cache-Friendly Data Layout

- Row-major tensor storage for better cache locality
- Contiguous memory allocation
- Stride calculations for efficient access

### 4. Lazy Evaluation Opportunities

Views enable lazy evaluation:

```cpp
// No data copying until explicitly needed
auto slice = tensor.slice(dim, index);
auto result = slice + other_tensor;  // Compute on demand
```

### 5. Compiler Optimizations

The library leverages:
- Inline functions for small operations
- Constexpr for compile-time computations
- Move semantics for efficient data transfer
- RVO (Return Value Optimization)

**Compilation Flags:**
```cmake
# Release build with optimizations
-O3 -march=native
```

### 6. Minimizing Allocations

- Reuse tensors where possible
- In-place operations (operator+=, operator*=, etc.)
- Pre-allocation of workspace tensors

---

## Extension Points

### Adding a New Layer

1. Inherit from `Layer<T>`
2. Implement `forward()` and `backward()`
3. Override `updateWeights()` if the layer has parameters
4. Override `setTrainingMode()` if behavior differs between training/eval

```cpp
template <typename T = float>
class MyCustomLayer : public Layer<T> {
public:
    MyCustomLayer(/* parameters */) { /* initialize */ }
    
    Tensor<T> forward(const Tensor<T>& input) override {
        // Forward computation
        return output;
    }
    
    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        // Backward computation
        return gradInput;
    }
    
    void updateWeights(Optimizer<T>& optimizer) override {
        // Update parameters if any
    }
};
```

### Adding a New Optimizer

1. Inherit from `Optimizer<T>`
2. Implement `optimize()` method
3. Maintain optimizer state as member variables

```cpp
template <typename T = float>
class MyOptimizer : public Optimizer<T> {
public:
    void optimize(
        const std::vector<std::reference_wrapper<Tensor<T>>>& weights,
        const std::vector<std::reference_wrapper<Tensor<T>>>& gradients,
        T learningRateOverride = T(-1)) override {
        
        for (size_t i = 0; i < weights.size(); ++i) {
            // Update logic
            weights[i].get() -= learningRate * gradients[i].get();
        }
    }
    
private:
    T learningRate;
    // Other optimizer state
};
```

### Adding a New Loss Function

1. Inherit from `Loss<T>`
2. Implement `compute()` for loss calculation
3. Implement `gradient()` for gradient computation

```cpp
template <typename T = float>
class MyLoss : public Loss<T> {
public:
    Tensor<T> compute(const Tensor<T>& prediction, 
                      const Tensor<T>& target) override {
        // Compute loss value
        return lossValue;
    }
    
    Tensor<T> gradient(const Tensor<T>& prediction, 
                       const Tensor<T>& target) override {
        // Compute gradient of loss
        return gradLoss;
    }
};
```

### Adding a New Activation Function

1. Inherit from `Activation<T>`
2. Implement `activate()` for forward pass
3. Implement `derivative()` for backward pass

```cpp
template <typename T = float>
class MyActivation : public Activation<T> {
public:
    Tensor<T> activate(const Tensor<T>& input) const override {
        // Apply activation
        return activated;
    }
    
    Tensor<T> derivative(const Tensor<T>& input, 
                         const Tensor<T>& output) const override {
        // Compute derivative
        return grad;
    }
};
```

---

## Directory Structure

```
smart_dnn/
├── SmartDNN.hpp              # Main model class header
├── SmartDNN/
│   └── SmartDNN.impl.hpp     # Implementation
├── Layer.hpp                 # Base layer interface
├── Layers/                   # Layer implementations
│   ├── FullyConnectedLayer.hpp
│   ├── Conv2DLayer.hpp
│   ├── ActivationLayer.hpp
│   └── FlattenLayer.hpp
├── Activation.hpp            # Base activation interface
├── Activations/              # Activation implementations
│   ├── ReLU.hpp
│   ├── Sigmoid.hpp
│   ├── Tanh.hpp
│   └── Softmax.hpp
├── Optimizer.hpp             # Base optimizer interface
├── Optimizers/               # Optimizer implementations
│   ├── AdamOptimizer.hpp
│   ├── SGDOptimizer.hpp
│   └── RMSPropOptimizer.hpp
├── Loss.hpp                  # Base loss interface
├── Loss/                     # Loss implementations
│   ├── MSELoss.hpp
│   └── CategoricalCrossEntropyLoss.hpp
├── Tensor/                   # Tensor subsystem
│   ├── Tensor.hpp
│   ├── Tensor.impl.hpp
│   ├── TensorData.hpp
│   ├── TensorOperations.hpp
│   ├── AdvancedTensorOperations.hpp
│   ├── SliceView.hpp
│   └── BroadcastView.hpp
├── Shape/                    # Shape utilities
│   ├── Shape.hpp
│   └── ShapeOperations.hpp
├── Regularisation/           # Regularization layers
│   ├── DropoutLayer.hpp
│   ├── BatchNormalizationLayer.hpp
│   └── MaxPooling2DLayer.hpp
├── Datasets/                 # Data loading utilities
│   ├── MNistLoader.hpp
│   └── SampleGenerator.hpp
├── Debugging/                # Debugging utilities
│   └── Logger.hpp
└── RandomEngine.hpp          # Random number generation
```

---

## Design Patterns Used

### 1. Template Method Pattern

Base classes define the algorithm structure; subclasses implement specific steps.

```cpp
class SmartDNN {
    void train() {
        for (epoch in epochs) {
            forward();    // Implemented by layers
            backward();   // Implemented by layers
            update();     // Implemented by optimizer
        }
    }
};
```

### 2. Strategy Pattern

Interchangeable algorithms (loss functions, optimizers).

```cpp
model.compile(MSELoss(), AdamOptimizer());  // Strategy 1
model.compile(CategoricalCrossEntropyLoss(), SGDOptimizer());  // Strategy 2
```

### 3. Composite Pattern

Layers are composed to form a network.

```cpp
model.addLayer(layer1);
model.addLayer(layer2);
model.addLayer(layer3);
```

### 4. Factory Pattern

Static factory methods for tensor creation.

```cpp
auto zeros = Tensor<float>::zeros(shape);
auto ones = Tensor<float>::ones(shape);
auto random = Tensor<float>::rand(shape);
```

---

## Thread Safety

**Current Status**: SmartDNN is **not thread-safe** by default.

**Recommendations:**
- Use separate model instances per thread
- External synchronization if sharing models
- Data loading can be parallelized independently

**Future Work**: Thread-safe operations and parallel training support.

---

## Future Architecture Enhancements

1. **GPU Support**: CUDA backend for TensorData
2. **Graph Computation**: Build computation graphs for optimization
3. **Automatic Differentiation**: More flexible gradient computation
4. **Mixed Precision**: FP16 support for faster training
5. **Distributed Training**: Multi-GPU and multi-node support
6. **Model Zoo**: Pre-trained models and architectures
7. **ONNX Export**: Interoperability with other frameworks

---

## Conclusion

SmartDNN's architecture emphasizes simplicity, performance, and extensibility. The modular design makes it easy to understand, extend, and maintain while delivering high performance through modern C++ techniques.

For implementation details, see the [API Reference](API_REFERENCE.md).
For usage examples, see the [Tutorials](TUTORIALS.md).

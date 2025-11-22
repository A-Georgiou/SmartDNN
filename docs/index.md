# SmartDNN Documentation

Welcome to the SmartDNN documentation! SmartDNN is a high-performance C++ deep learning library designed for flexibility, efficiency, and ease of use.

## 📚 Documentation Overview

This documentation is organized into several guides to help you get the most out of SmartDNN:

### [API Reference](API_REFERENCE.md)
Complete API documentation for all classes, functions, and components in SmartDNN.

**What you'll find:**
- Core classes (SmartDNN, Tensor, Layer, Loss, Optimizer)
- Layer types (FullyConnected, Conv2D, Activation, etc.)
- Activation functions (ReLU, Sigmoid, Tanh, Softmax, etc.)
- Optimizers (Adam, SGD, RMSProp)
- Loss functions (MSE, Categorical Cross Entropy)
- Utilities (Shape, data loaders)

**Best for:** Looking up specific API details, method signatures, and parameters.

---

### [Architecture Guide](ARCHITECTURE.md)
In-depth overview of SmartDNN's internal architecture and design principles.

**What you'll find:**
- Design philosophy and patterns
- Component hierarchy and relationships
- Data flow during training and inference
- Memory management strategies
- Performance optimizations
- Extension points for custom components

**Best for:** Understanding how SmartDNN works internally, extending the library, or contributing to development.

---

### [Tutorials](TUTORIALS.md)
Step-by-step tutorials from beginner to advanced topics.

**What you'll find:**
- Getting started guide
- Tutorial 1: Your first neural network
- Tutorial 2: Linear regression
- Tutorial 3: Multi-layer perceptron for classification
- Tutorial 4: Convolutional neural networks (CNNs)
- Tutorial 5: MNIST digit recognition
- Tutorial 6: Advanced training techniques
- Tutorial 7: Model persistence (saving/loading)
- Tutorial 8: Custom layers

**Best for:** Learning by doing, building your first models, and exploring features.

---

### [Contributing Guide](CONTRIBUTING.md)
Everything you need to know about contributing to SmartDNN.

**What you'll find:**
- Development setup
- Coding standards and style guide
- Pull request process
- Testing guidelines
- Documentation standards
- How to report bugs and suggest features

**Best for:** Contributors who want to help improve SmartDNN.

---

### [Performance Guide](PERFORMANCE.md)
Best practices and techniques for optimizing performance.

**What you'll find:**
- Compilation optimizations
- Model architecture optimization
- Training optimizations
- Memory management strategies
- Benchmarking and profiling
- Platform-specific optimizations
- Common performance pitfalls

**Best for:** Maximizing training speed and inference performance.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/A-Georgiou/SmartDNN.git
cd SmartDNN

# Build with CMake
cmake .
make
```

### Your First Model

```cpp
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Loss/MSELoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"

using namespace smart_dnn;

int main() {
    // Create model
    SmartDNN<float> model;
    
    // Add layers
    model.addLayer(FullyConnectedLayer(10, 100));
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(FullyConnectedLayer(100, 10));
    
    // Compile
    AdamOptions options;
    options.learningRate = 0.001f;
    model.compile(MSELoss(), AdamOptimizer(options));
    
    // Train
    model.train(inputs, targets, 100);
    
    // Predict
    model.evalMode();
    auto prediction = model.predict(testInput);
    
    return 0;
}
```

See the [Tutorials](TUTORIALS.md) for more detailed examples.

---

## 📖 Key Concepts

### Tensors
Multi-dimensional arrays that hold your data. All operations in SmartDNN work with tensors.

```cpp
Tensor<float> tensor(Shape{3, 3}, 1.0f);  // 3x3 tensor filled with 1.0
auto zeros = Tensor<float>::zeros(Shape{2, 2});
auto random = Tensor<float>::rand(Shape{5, 5});
```

### Layers
Building blocks of neural networks. Stack layers to create your model.

```cpp
model.addLayer(FullyConnectedLayer(128, 64));  // Dense layer
model.addLayer(Conv2DLayer(32, 64, 3));        // Convolutional layer
model.addLayer(ActivationLayer(ReLU()));       // Activation layer
model.addLayer(DropoutLayer(0.5f));            // Regularization
```

### Training Loop
SmartDNN handles the training loop for you:

1. **Forward pass**: Compute predictions
2. **Loss computation**: Measure error
3. **Backward pass**: Compute gradients
4. **Weight update**: Optimize parameters

```cpp
model.train(inputs, targets, epochs);
```

### Inference
After training, use the model for predictions:

```cpp
model.evalMode();  // Disable dropout, use batch norm running stats
auto prediction = model.predict(input);
```

---

## 🎯 Common Use Cases

### Linear Regression
```cpp
model.addLayer(FullyConnectedLayer(1, 10));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(FullyConnectedLayer(10, 1));
model.compile(MSELoss(), AdamOptimizer());
```

### Binary Classification
```cpp
model.addLayer(FullyConnectedLayer(features, 64));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(FullyConnectedLayer(64, 1));
model.addLayer(ActivationLayer(Sigmoid()));
model.compile(MSELoss(), AdamOptimizer());
```

### Multi-class Classification
```cpp
model.addLayer(FullyConnectedLayer(features, 128));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(FullyConnectedLayer(128, num_classes));
model.addLayer(ActivationLayer(Softmax()));
model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer());
```

### Image Classification (CNN)
```cpp
model.addLayer(Conv2DLayer(1, 32, 3));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(MaxPooling2DLayer(2, 2));
model.addLayer(FlattenLayer());
model.addLayer(FullyConnectedLayer(flat_size, 128));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(FullyConnectedLayer(128, num_classes));
model.addLayer(ActivationLayer(Softmax()));
model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer());
```

---

## 🔧 Advanced Topics

### Custom Layers
Create your own layer by inheriting from `Layer<T>`:

```cpp
template <typename T = float>
class MyCustomLayer : public Layer<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) override {
        // Your forward logic
        return output;
    }
    
    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        // Your backward logic
        return gradInput;
    }
};
```

See [Tutorial 8](TUTORIALS.md#tutorial-8-custom-layers) for details.

### Model Persistence

```cpp
// Save model
model.saveModel("my_model.bin");

// Load model
model.loadModel("my_model.bin");
```

### Regularization Techniques

```cpp
// Dropout
model.addLayer(DropoutLayer(0.5f));

// Batch Normalization
model.addLayer(BatchNormalizationLayer(num_features));

// L2 Regularization (in optimizer)
options.l2Strength = 0.0001f;
```

---

## 📊 Performance

SmartDNN is optimized for high performance:

- **53% faster** than non-templated version for linear regression
- **99.8% faster** than non-templated version for MNIST classification

Key optimizations:
- Template-based design for zero-cost abstractions
- Move semantics to reduce copying
- Cache-friendly memory layout
- Parallel operations with OpenMP
- Aggressive compiler optimizations

See the [Performance Guide](PERFORMANCE.md) for optimization tips.

---

## 🤝 Getting Help

### Documentation
- **API Reference**: Look up specific functions and classes
- **Tutorials**: Learn by example
- **Architecture Guide**: Understand internal design

### Community
- **GitHub Issues**: Report bugs or request features
- **Email**: AndrewGeorgiou98@outlook.com
- **Examples**: Check the `examples/` directory

### Contributing
We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md) for details.

---

## 📝 License

SmartDNN is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

---

## 🗺️ Roadmap

Future enhancements planned:
- GPU acceleration with CUDA
- Extended layer support (LSTM, GRU, Attention)
- Model zoo with pre-trained models
- ONNX export for interoperability
- Distributed training support
- Mixed precision training

---

## 📚 Additional Resources

### Examples
- [Simple Linear Regression](../examples/SimpleLinearRegressionModel.cpp)
- [MNIST CNN](../examples/MNistModel.cpp)

### External Resources
- [C++ Reference](https://en.cppreference.com/)
- [Neural Networks Basics](https://www.deeplearningbook.org/)
- [Modern C++ Features](https://github.com/AnthonyCalandra/modern-cpp-features)

---

## 🎓 Learning Path

**Beginner:**
1. Read the [Quick Start](#quick-start) section
2. Follow [Tutorial 1](TUTORIALS.md#tutorial-1-your-first-neural-network)
3. Experiment with [Tutorial 2](TUTORIALS.md#tutorial-2-linear-regression)

**Intermediate:**
1. Build a [classifier](TUTORIALS.md#tutorial-3-multi-layer-perceptron-for-classification)
2. Learn about [CNNs](TUTORIALS.md#tutorial-4-convolutional-neural-network-cnn)
3. Train on [MNIST](TUTORIALS.md#tutorial-5-mnist-digit-recognition)

**Advanced:**
1. Study the [Architecture Guide](ARCHITECTURE.md)
2. Create [custom layers](TUTORIALS.md#tutorial-8-custom-layers)
3. Optimize [performance](PERFORMANCE.md)
4. [Contribute](CONTRIBUTING.md) to the project

---

## 📮 Contact

For questions, suggestions, or contributions:
- **Email**: AndrewGeorgiou98@outlook.com
- **GitHub**: [A-Georgiou/SmartDNN](https://github.com/A-Georgiou/SmartDNN)

---

**Happy Deep Learning with SmartDNN! 🚀**

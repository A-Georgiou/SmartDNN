# SmartDNN Tutorials

This guide provides step-by-step tutorials for using SmartDNN, from basic concepts to advanced techniques.

## Table of Contents

- [Getting Started](#getting-started)
- [Tutorial 1: Your First Neural Network](#tutorial-1-your-first-neural-network)
- [Tutorial 2: Linear Regression](#tutorial-2-linear-regression)
- [Tutorial 3: Multi-Layer Perceptron for Classification](#tutorial-3-multi-layer-perceptron-for-classification)
- [Tutorial 4: Convolutional Neural Network (CNN)](#tutorial-4-convolutional-neural-network-cnn)
- [Tutorial 5: MNIST Digit Recognition](#tutorial-5-mnist-digit-recognition)
- [Tutorial 6: Advanced Training Techniques](#tutorial-6-advanced-training-techniques)
- [Tutorial 7: Model Persistence](#tutorial-7-model-persistence)
- [Tutorial 8: Custom Layers](#tutorial-8-custom-layers)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10 or higher
- Basic understanding of C++ and neural networks

### Installation

#### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/A-Georgiou/SmartDNN.git
cd SmartDNN

# Configure with CMake
cmake .

# Build (this creates the executable)
make
```

#### Option 2: Docker Installation

```bash
# Clone the repository
git clone https://github.com/A-Georgiou/SmartDNN.git
cd SmartDNN

# Build Docker image
docker build -f .docker/Dockerfile -t smartdnn-app .

# Run the container
docker run --rm -it smartdnn-app
```

### Project Structure for Your Code

Create a `src/main.cpp` file in the SmartDNN directory:

```bash
mkdir -p src
touch src/main.cpp
```

Then build and run:

```bash
make
./SmartDNN
```

---

## Tutorial 1: Your First Neural Network

Let's create a simple neural network that learns to approximate a function.

### Step 1: Include Headers

```cpp
#include <iostream>
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Loss/MSELoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"

using namespace smart_dnn;
```

### Step 2: Create Training Data

```cpp
int main() {
    // Create simple dataset: y = 2x + 1
    std::vector<Tensor<float>> inputs;
    std::vector<Tensor<float>> targets;
    
    for (float x = 0.0f; x < 10.0f; x += 0.5f) {
        inputs.push_back(Tensor<float>(Shape{1}, x));
        targets.push_back(Tensor<float>(Shape{1}, 2.0f * x + 1.0f));
    }
    
    std::cout << "Created " << inputs.size() << " training samples" << std::endl;
```

### Step 3: Build the Model

```cpp
    // Create model
    SmartDNN<float> model;
    
    // Add layers
    model.addLayer(FullyConnectedLayer(1, 10));  // Input layer: 1 -> 10
    model.addLayer(ActivationLayer(ReLU()));      // ReLU activation
    model.addLayer(FullyConnectedLayer(10, 1));   // Output layer: 10 -> 1
    
    std::cout << "Model created with 2 layers" << std::endl;
```

### Step 4: Compile and Train

```cpp
    // Configure optimizer
    AdamOptions options;
    options.learningRate = 0.01f;
    
    // Compile model
    model.compile(MSELoss(), AdamOptimizer(options));
    
    // Train
    std::cout << "Training..." << std::endl;
    model.train(inputs, targets, 100);  // 100 epochs
```

### Step 5: Make Predictions

```cpp
    // Test the model
    model.evalMode();
    
    Tensor<float> testInput(Shape{1}, 5.0f);
    Tensor<float> prediction = model.predict(testInput);
    
    std::cout << "Input: 5.0, Prediction: " << prediction.toDetailedString() 
              << ", Expected: 11.0" << std::endl;
    
    return 0;
}
```

### Complete Code

Save this as `src/main.cpp`:

```cpp
#include <iostream>
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Loss/MSELoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"

using namespace smart_dnn;

int main() {
    // Create dataset
    std::vector<Tensor<float>> inputs;
    std::vector<Tensor<float>> targets;
    
    for (float x = 0.0f; x < 10.0f; x += 0.5f) {
        inputs.push_back(Tensor<float>(Shape{1}, x));
        targets.push_back(Tensor<float>(Shape{1}, 2.0f * x + 1.0f));
    }
    
    // Build model
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer(1, 10));
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(FullyConnectedLayer(10, 1));
    
    // Compile
    AdamOptions options;
    options.learningRate = 0.01f;
    model.compile(MSELoss(), AdamOptimizer(options));
    
    // Train
    model.train(inputs, targets, 100);
    
    // Predict
    model.evalMode();
    Tensor<float> testInput(Shape{1}, 5.0f);
    Tensor<float> prediction = model.predict(testInput);
    
    std::cout << "Prediction for x=5: " << prediction.toDetailedString() 
              << " (expected: 11.0)" << std::endl;
    
    return 0;
}
```

Build and run:

```bash
make
./SmartDNN
```

---

## Tutorial 2: Linear Regression

A more complete linear regression example with noise and visualization.

### Complete Example

```cpp
#include <iostream>
#include <iomanip>
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Loss/MSELoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "smart_dnn/Datasets/SampleGenerator.hpp"

using namespace smart_dnn;

int main() {
    // Hyperparameters
    constexpr int BATCH_SIZE = 100;
    constexpr int EPOCHS = 100;
    constexpr float LEARNING_RATE = 0.01f;
    
    // Generate linear dataset: y = 2.0 * x + 3.0 + gaussian_noise
    // Note: BATCH_SIZE here means total number of samples in the dataset
    auto [inputs, targets] = generateLinearDataset(BATCH_SIZE);
    
    std::cout << "Generated " << inputs.size() 
              << " samples for linear regression" << std::endl;
    
    // Build model
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer(1, 10));
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(FullyConnectedLayer(10, 1));
    
    // Configure optimizer
    AdamOptions adamOptions;
    adamOptions.learningRate = LEARNING_RATE;
    model.compile(MSELoss(), AdamOptimizer(adamOptions));
    
    // Train
    std::cout << "Training for " << EPOCHS << " epochs..." << std::endl;
    model.train(inputs, targets, EPOCHS);
    
    // Evaluate
    model.evalMode();
    
    // Test on multiple points
    std::cout << "\nTest Results:" << std::endl;
    std::cout << std::setw(10) << "Input" 
              << std::setw(15) << "Prediction" 
              << std::setw(15) << "Expected" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (float x = 0.0f; x <= 10.0f; x += 2.0f) {
        Tensor<float> input(Shape{1}, x);
        Tensor<float> prediction = model.predict(input);
        float expected = 2.0f * x + 3.0f;
        
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << x
                  << std::setw(15) << prediction[0]
                  << std::setw(15) << expected << std::endl;
    }
    
    return 0;
}
```

**Expected Output:**
```
Generated 100 samples for linear regression
Training for 100 epochs...
Epoch 1/100, Loss: 45.234
...
Epoch 100/100, Loss: 0.123

Test Results:
     Input     Prediction       Expected
----------------------------------------
      0.00           3.05           3.00
      2.00           7.01           7.00
      4.00          11.03          11.00
      6.00          14.98          15.00
      8.00          18.97          19.00
     10.00          23.02          23.00
```

---

## Tutorial 3: Multi-Layer Perceptron for Classification

Building a classifier with multiple hidden layers.

### Problem: Binary Classification

```cpp
#include <iostream>
#include <cmath>
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Activations/Sigmoid.hpp"
#include "smart_dnn/Loss/MSELoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"

using namespace smart_dnn;

// Generate XOR dataset
std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> 
generateXORDataset() {
    std::vector<Tensor<float>> inputs;
    std::vector<Tensor<float>> targets;
    
    // XOR problem: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    std::vector<std::vector<float>> data = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f}
    };
    
    // Repeat dataset for training
    for (int i = 0; i < 100; i++) {
        for (auto& sample : data) {
            inputs.push_back(Tensor<float>(Shape{2}, {sample[0], sample[1]}));
            targets.push_back(Tensor<float>(Shape{1}, sample[2]));
        }
    }
    
    return {inputs, targets};
}

int main() {
    // Generate XOR dataset
    auto [inputs, targets] = generateXORDataset();
    
    // Build deeper network (XOR is not linearly separable)
    SmartDNN<float> model;
    model.addLayer(FullyConnectedLayer(2, 8));    // Input -> Hidden1
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(FullyConnectedLayer(8, 8));    // Hidden1 -> Hidden2
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(FullyConnectedLayer(8, 1));    // Hidden2 -> Output
    model.addLayer(ActivationLayer(Sigmoid()));    // Sigmoid for binary output
    
    // Train
    AdamOptions options;
    options.learningRate = 0.01f;
    model.compile(MSELoss(), AdamOptimizer(options));
    
    std::cout << "Training XOR classifier..." << std::endl;
    model.train(inputs, targets, 200);
    
    // Test
    model.evalMode();
    std::cout << "\nTest Results:" << std::endl;
    
    std::vector<std::vector<float>> testCases = {
        {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
    };
    
    for (auto& test : testCases) {
        Tensor<float> input(Shape{2}, test);
        Tensor<float> output = model.predict(input);
        
        std::cout << "Input: (" << test[0] << ", " << test[1] << ") "
                  << "-> Output: " << output[0] 
                  << " (rounded: " << std::round(output[0]) << ")"
                  << std::endl;
    }
    
    return 0;
}
```

---

## Tutorial 4: Convolutional Neural Network (CNN)

Building a CNN for image data processing.

### Basic CNN Architecture

```cpp
#include <iostream>
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/Conv2DLayer.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Layers/FlattenLayer.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Activations/Softmax.hpp"
#include "smart_dnn/Regularisation/MaxPooling2DLayer.hpp"
#include "smart_dnn/Regularisation/DropoutLayer.hpp"
#include "smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"

using namespace smart_dnn;

int main() {
    // Build CNN architecture
    SmartDNN<float> model;
    
    // First convolutional block
    model.addLayer(Conv2DLayer(1, 32, 3));       // 1 input channel, 32 filters, 3x3 kernel
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(MaxPooling2DLayer(2, 2));     // 2x2 pooling
    model.addLayer(DropoutLayer(0.25f));         // 25% dropout
    
    // Second convolutional block
    model.addLayer(Conv2DLayer(32, 64, 3));      // 32 input channels, 64 filters
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(MaxPooling2DLayer(2, 2));
    model.addLayer(DropoutLayer(0.25f));
    
    // Fully connected layers
    model.addLayer(FlattenLayer());
    model.addLayer(FullyConnectedLayer(1600, 128));  // Adjust based on input size
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(DropoutLayer(0.5f));
    
    // Output layer
    model.addLayer(FullyConnectedLayer(128, 10));    // 10 classes
    model.addLayer(ActivationLayer(Softmax()));
    
    // Configure optimizer
    AdamOptions options;
    options.learningRate = 0.001f;
    options.beta1 = 0.9f;
    options.beta2 = 0.999f;
    
    // Compile
    model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer(options));
    
    std::cout << "CNN model built successfully" << std::endl;
    std::cout << "Architecture:" << std::endl;
    std::cout << "  Conv2D(1->32, 3x3) -> ReLU -> MaxPool(2x2) -> Dropout(0.25)" << std::endl;
    std::cout << "  Conv2D(32->64, 3x3) -> ReLU -> MaxPool(2x2) -> Dropout(0.25)" << std::endl;
    std::cout << "  Flatten -> FC(1600->128) -> ReLU -> Dropout(0.5)" << std::endl;
    std::cout << "  FC(128->10) -> Softmax" << std::endl;
    
    return 0;
}
```

---

## Tutorial 5: MNIST Digit Recognition

Complete example for training on MNIST dataset.

### Step 1: Download MNIST Dataset

```bash
# Create dataset directory
mkdir -p .datasets
cd .datasets

# Download MNIST files
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Decompress
gunzip *.gz
cd ..
```

### Step 2: Complete MNIST Training Code

```cpp
#include <iostream>
#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Layers/Conv2DLayer.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Layers/FlattenLayer.hpp"
#include "smart_dnn/Activations/ReLU.hpp"
#include "smart_dnn/Activations/Softmax.hpp"
#include "smart_dnn/Regularisation/MaxPooling2DLayer.hpp"
#include "smart_dnn/Regularisation/DropoutLayer.hpp"
#include "smart_dnn/Regularisation/BatchNormalizationLayer.hpp"
#include "smart_dnn/Loss/CategoricalCrossEntropyLoss.hpp"
#include "smart_dnn/Optimizers/AdamOptimizer.hpp"
#include "smart_dnn/Datasets/MNistLoader.hpp"

using namespace smart_dnn;

int main() {
    // Hyperparameters
    constexpr int EPOCHS = 10;
    constexpr int BATCH_SIZE = 8;
    constexpr int SAMPLE_COUNT = 1000;  // Use subset for faster training
    constexpr float LEARNING_RATE = 0.001f;
    
    // Build model
    SmartDNN<float> model;
    
    // Convolutional layers
    model.addLayer(Conv2DLayer(1, 32, 3));
    model.addLayer(BatchNormalizationLayer(32));
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(MaxPooling2DLayer(2, 2));
    model.addLayer(DropoutLayer(0.25f));
    
    // Flatten and fully connected
    model.addLayer(FlattenLayer());
    model.addLayer(FullyConnectedLayer(5408, 128));  // 32*13*13 = 5408
    model.addLayer(BatchNormalizationLayer(128));
    model.addLayer(ActivationLayer(ReLU()));
    model.addLayer(DropoutLayer(0.25f));
    
    // Output layer
    model.addLayer(FullyConnectedLayer(128, 10));
    model.addLayer(ActivationLayer(Softmax()));
    
    // Configure optimizer
    AdamOptions adamOptions;
    adamOptions.learningRate = LEARNING_RATE;
    adamOptions.beta1 = 0.9f;
    adamOptions.beta2 = 0.999f;
    model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer(adamOptions));
    
    // Load data
    std::cout << "Loading MNIST dataset..." << std::endl;
    std::string imagesPath = ".datasets/train-images-idx3-ubyte";
    std::string labelsPath = ".datasets/train-labels-idx1-ubyte";
    
    MNISTLoader dataLoader(imagesPath, labelsPath, BATCH_SIZE, SAMPLE_COUNT);
    auto [inputs, targets] = dataLoader.loadData();
    
    std::cout << "Loaded " << inputs.size() << " samples" << std::endl;
    
    // Train
    std::cout << "Training model..." << std::endl;
    model.train(inputs, targets, EPOCHS);
    
    // Evaluate
    model.evalMode();
    
    // Test on a few samples
    std::cout << "\nTest Predictions:" << std::endl;
    for (int i = 0; i < 5; i++) {
        Tensor<float> input = inputs[i];
        Tensor<float> target = targets[i];
        Tensor<float> prediction = model.predict(input);
        
        // Display as ASCII art
        std::cout << "\nSample " << i+1 << ":" << std::endl;
        std::cout << dataLoader.toAsciiArt(input) << std::endl;
        
        // Find predicted class
        int predictedClass = 0;
        float maxProb = prediction[0];
        for (int j = 1; j < 10; j++) {
            if (prediction[j] > maxProb) {
                maxProb = prediction[j];
                predictedClass = j;
            }
        }
        
        // Find actual class
        int actualClass = 0;
        for (int j = 0; j < 10; j++) {
            if (target[j] > 0.5f) {
                actualClass = j;
                break;
            }
        }
        
        std::cout << "Predicted: " << predictedClass 
                  << " (confidence: " << maxProb << ")" << std::endl;
        std::cout << "Actual: " << actualClass << std::endl;
    }
    
    return 0;
}
```

---

## Tutorial 6: Advanced Training Techniques

### Technique 1: Learning Rate Scheduling

```cpp
// Manually adjust learning rate during training
AdamOptions options;
options.learningRate = 0.01f;
model.compile(MSELoss(), AdamOptimizer(options));

// Train for initial epochs
model.train(inputs, targets, 50);

// Reduce learning rate
options.learningRate = 0.001f;
model.compile(MSELoss(), AdamOptimizer(options));
model.train(inputs, targets, 50);
```

### Technique 2: Regularization

```cpp
// Use L2 regularization
AdamOptions options;
options.learningRate = 0.001f;
options.l2Strength = 0.0001f;  // L2 regularization
model.compile(MSELoss(), AdamOptimizer(options));
```

### Technique 3: Batch Normalization

```cpp
#include "smart_dnn/Regularisation/BatchNormalizationLayer.hpp"

// Add batch normalization after layers
model.addLayer(FullyConnectedLayer(128, 64));
model.addLayer(BatchNormalizationLayer(64));  // Normalize 64 features
model.addLayer(ActivationLayer(ReLU()));
```

### Technique 4: Dropout for Regularization

```cpp
// Different dropout rates for different layers
model.addLayer(FullyConnectedLayer(256, 128));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(DropoutLayer(0.3f));  // 30% dropout

model.addLayer(FullyConnectedLayer(128, 64));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(DropoutLayer(0.5f));  // 50% dropout (more aggressive)
```

### Technique 5: Different Optimizers

```cpp
#include "smart_dnn/Optimizers/SGDOptimizer.hpp"
#include "smart_dnn/Optimizers/RMSPropOptimizer.hpp"

// SGD with momentum
model.compile(MSELoss(), SGDOptimizer(0.01f, 0.9f));

// Or RMSProp
model.compile(MSELoss(), RMSPropOptimizer(0.001f, 0.9f));

// Or Adam (recommended for most cases)
AdamOptions options;
options.learningRate = 0.001f;
model.compile(MSELoss(), AdamOptimizer(options));
```

---

## Tutorial 7: Model Persistence

### Saving a Model

```cpp
#include <iostream>
#include "smart_dnn/SmartDNN.hpp"

int main() {
    SmartDNN<float> model;
    
    // Build and train your model
    // ... (model architecture and training code)
    
    // Save the trained model
    std::cout << "Saving model..." << std::endl;
    model.saveModel("my_trained_model.bin");
    std::cout << "Model saved successfully!" << std::endl;
    
    return 0;
}
```

### Loading a Model

```cpp
#include <iostream>
#include "smart_dnn/SmartDNN.hpp"

int main() {
    SmartDNN<float> model;
    
    // Build the same architecture as the saved model
    // ... (same architecture as when saving)
    
    // Load the trained weights
    std::cout << "Loading model..." << std::endl;
    model.loadModel("my_trained_model.bin");
    std::cout << "Model loaded successfully!" << std::endl;
    
    // Use the model for inference
    model.evalMode();
    
    Tensor<float> input(Shape{1, 28, 28});
    Tensor<float> prediction = model.predict(input);
    
    return 0;
}
```

**Important Notes:**
- Model architecture must match when loading
- Only weights and optimizer state are saved
- The architecture must be reconstructed in code

---

## Tutorial 8: Custom Layers

### Creating a Custom Layer

```cpp
#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Tensor/Tensor.hpp"

namespace smart_dnn {

template <typename T = float>
class CustomSquareLayer : public Layer<T> {
public:
    CustomSquareLayer() = default;
    
    // Forward pass: square each element
    Tensor<T> forward(const Tensor<T>& input) override {
        this->input = input;
        
        // Create output tensor
        Tensor<T> output = input;
        
        // Square each element
        output.apply([](T x) { return x * x; });
        
        return output;
    }
    
    // Backward pass: gradient is 2*input
    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        Tensor<T> gradInput = input;
        
        // Gradient: d/dx(x^2) = 2x
        gradInput.apply([](T x) { return 2 * x; });
        
        // Chain rule: multiply by upstream gradient
        gradInput = gradInput * gradOutput;
        
        return gradInput;
    }

private:
    Tensor<T> input;  // Store input for backward pass
};

} // namespace smart_dnn
```

### Using the Custom Layer

```cpp
#include "CustomSquareLayer.hpp"

int main() {
    SmartDNN<float> model;
    
    model.addLayer(FullyConnectedLayer(10, 20));
    model.addLayer(CustomSquareLayer());  // Use custom layer
    model.addLayer(FullyConnectedLayer(20, 10));
    
    // Train as usual
    // ...
    
    return 0;
}
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Compilation Errors

**Problem:** "Cannot find header files"

**Solution:**
```bash
# Make sure you're in the SmartDNN directory
cd /path/to/SmartDNN

# Include path should point to the root directory
# CMakeLists.txt should have:
include_directories(${PROJECT_SOURCE_DIR})
```

#### 2. Shape Mismatch Errors

**Problem:** Runtime error about incompatible tensor shapes

**Solution:**
```cpp
// Check tensor shapes
std::cout << "Tensor shape: " << tensor.getShape().toString() << std::endl;

// Ensure output of one layer matches input of next
// For example, after Conv2D and MaxPooling:
// Output size = (input_size - kernel_size) / stride + 1
```

#### 3. NaN or Inf in Training

**Problem:** Loss becomes NaN during training

**Solutions:**
```cpp
// 1. Reduce learning rate
options.learningRate = 0.0001f;  // Try smaller value

// 2. Use gradient clipping (if available)
// 3. Check for division by zero
// 4. Use batch normalization
model.addLayer(BatchNormalizationLayer(numFeatures));

// 5. Initialize weights properly (already done by SmartDNN)
```

#### 4. Slow Training

**Solutions:**
```cpp
// 1. Use batch processing (already supported)
// 2. Reduce model size
// 3. Use Release build mode
cmake -DCMAKE_BUILD_TYPE=Release .

// 4. Reduce sample count for testing
MNISTLoader loader(path1, path2, batchSize, 1000);  // Use 1000 samples
```

#### 5. Poor Accuracy

**Solutions:**
```cpp
// 1. Train for more epochs
model.train(inputs, targets, 200);  // Increase epochs

// 2. Adjust learning rate
options.learningRate = 0.001f;  // Tune this value

// 3. Add more layers or neurons
model.addLayer(FullyConnectedLayer(128, 256));  // Increase capacity

// 4. Use regularization
model.addLayer(DropoutLayer(0.3f));
options.l2Strength = 0.0001f;

// 5. Normalize input data
// Make sure inputs are in range [0, 1] or [-1, 1]
```

---

## Best Practices

### 1. Data Preparation

```cpp
// Normalize inputs to [0, 1] range
for (auto& tensor : inputs) {
    tensor = tensor / 255.0f;  // For image data
}

// Or standardize to mean=0, std=1
// data = (data - mean) / std
```

### 2. Model Architecture

```cpp
// Start simple, then add complexity
// Bad: Immediately building complex architecture
// Good: Start with baseline, then improve

// Baseline
model.addLayer(FullyConnectedLayer(input_size, output_size));

// Then add hidden layers
model.addLayer(FullyConnectedLayer(input_size, 128));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(FullyConnectedLayer(128, output_size));

// Then add regularization
model.addLayer(DropoutLayer(0.3f));
```

### 3. Training Strategy

```cpp
// 1. Use training/validation split
// 2. Monitor loss during training
// 3. Use evalMode() for validation
// 4. Save best model

model.trainingMode();
model.train(train_inputs, train_targets, epochs);

model.evalMode();
auto val_predictions = model.predict(val_inputs);
// Compute validation loss
```

### 4. Hyperparameter Tuning

Try values in this order:
1. Learning rate: [0.1, 0.01, 0.001, 0.0001]
2. Batch size: [8, 16, 32, 64]
3. Hidden layer sizes: [64, 128, 256, 512]
4. Dropout rate: [0.2, 0.3, 0.5]

---

## Next Steps

1. Experiment with different architectures
2. Try different datasets
3. Implement custom layers for specific tasks
4. Explore advanced optimization techniques
5. Contribute to the SmartDNN project!

For more information:
- [API Reference](API_REFERENCE.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Contributing Guide](CONTRIBUTING.md)

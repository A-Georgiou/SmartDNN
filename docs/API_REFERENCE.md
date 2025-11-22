# SmartDNN API Reference

This document provides a comprehensive reference for all classes, functions, and components in the SmartDNN library.

## Table of Contents

- [Core Classes](#core-classes)
  - [SmartDNN](#smartdnn)
  - [Tensor](#tensor)
  - [Layer](#layer)
  - [Loss](#loss)
  - [Optimizer](#optimizer)
- [Layers](#layers)
  - [FullyConnectedLayer](#fullyconnectedlayer)
  - [Conv2DLayer](#conv2dlayer)
  - [ActivationLayer](#activationlayer)
  - [FlattenLayer](#flattenlayer)
- [Regularization](#regularization)
  - [DropoutLayer](#dropoutlayer)
  - [BatchNormalizationLayer](#batchnormalizationlayer)
  - [MaxPooling2DLayer](#maxpooling2dlayer)
- [Activation Functions](#activation-functions)
  - [ReLU](#relu)
  - [Sigmoid](#sigmoid)
  - [Tanh](#tanh)
  - [Softmax](#softmax)
  - [LeakyReLU](#leakyrelu)
  - [Swish](#swish)
  - [Mish](#mish)
- [Optimizers](#optimizers)
  - [AdamOptimizer](#adamoptimizer)
  - [SGDOptimizer](#sgdoptimizer)
  - [RMSPropOptimizer](#rmspropoptimizer)
- [Loss Functions](#loss-functions)
  - [MSELoss](#mseloss)
  - [CategoricalCrossEntropyLoss](#categoricalcrossentropyloss)
- [Utilities](#utilities)
  - [Shape](#shape)
  - [MNISTLoader](#mnistloader)
  - [SampleGenerator](#samplegenerator)

---

## Core Classes

### SmartDNN

The main neural network model class that orchestrates layers, loss functions, and optimizers.

**Template Parameters:**
- `T` - Data type for computations (default: `float`)

**Constructor:**
```cpp
SmartDNN<T>()
```

**Methods:**

#### addLayer
```cpp
template<typename LayerType>
void addLayer(LayerType&& layer)
```
Adds a layer to the neural network model.

**Parameters:**
- `layer` - A layer object (e.g., FullyConnectedLayer, Conv2DLayer, ActivationLayer)

**Example:**
```cpp
SmartDNN<float> model;
model.addLayer(FullyConnectedLayer(10, 100));
model.addLayer(ActivationLayer(ReLU()));
```

#### compile
```cpp
template<typename LossType, typename OptimizerType>
void compile(LossType&& loss, OptimizerType&& optimizer)
```
Compiles the model with a loss function and optimizer.

**Parameters:**
- `loss` - Loss function object (e.g., MSELoss, CategoricalCrossEntropyLoss)
- `optimizer` - Optimizer object (e.g., AdamOptimizer, SGDOptimizer)

**Example:**
```cpp
AdamOptions options;
options.learningRate = 0.001f;
model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer(options));
```

#### train
```cpp
void train(const std::vector<Tensor<T>>& inputs, 
           const std::vector<Tensor<T>>& targets, 
           int epochs)
```
Trains the model on provided data.

**Parameters:**
- `inputs` - Vector of input tensors
- `targets` - Vector of target tensors
- `epochs` - Number of training epochs

**Example:**
```cpp
model.train(trainingInputs, trainingTargets, 100);
```

#### predict
```cpp
Tensor<T> predict(const Tensor<T>& input)
std::vector<Tensor<T>> predict(const std::vector<Tensor<T>>& inputs)
```
Makes predictions on input data.

**Parameters:**
- `input` - Single input tensor or vector of input tensors

**Returns:**
- Predicted output tensor(s)

**Example:**
```cpp
Tensor<float> input(Shape{1, 28, 28});
Tensor<float> prediction = model.predict(input);
```

#### trainingMode / evalMode
```cpp
void trainingMode()
void evalMode()
```
Sets the model to training or evaluation mode. This affects layers like Dropout and BatchNormalization.

**Example:**
```cpp
model.trainingMode();  // Enable dropout, batch norm training mode
model.train(inputs, targets, epochs);
model.evalMode();      // Disable dropout, use batch norm running stats
auto prediction = model.predict(testInput);
```

#### saveModel / loadModel
```cpp
void saveModel(const std::string& filename) const
void loadModel(const std::string& filename)
```
Saves or loads model weights and optimizer state.

**Parameters:**
- `filename` - Path to save/load the model

**Example:**
```cpp
model.saveModel("my_model.bin");
model.loadModel("my_model.bin");
```

---

### Tensor

Multi-dimensional array class for storing and manipulating data.

**Template Parameters:**
- `T` - Data type (default: `float`)
- `DeviceType` - Device type for computation (default: `CPUDevice`)

**Constructors:**
```cpp
Tensor(Shape dimensions)
Tensor(Shape dimensions, T value)
Tensor(Shape dimensions, const T* dataArray)
Tensor(Shape dimensions, std::initializer_list<T> values)
Tensor(Shape dimensions, const std::vector<T>& values)
```

**Example:**
```cpp
// Create a 3x3 tensor filled with zeros
Tensor<float> t1(Shape{3, 3});

// Create a 2x2 tensor filled with value 5.0
Tensor<float> t2(Shape{2, 2}, 5.0f);

// Create from initializer list
Tensor<float> t3(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
```

**Static Factory Methods:**

#### zeros
```cpp
static Tensor<T> zeros(Shape dimensions)
```
Creates a tensor filled with zeros.

#### ones
```cpp
static Tensor<T> ones(Shape dimensions)
```
Creates a tensor filled with ones.

#### rand
```cpp
static Tensor<T> rand(Shape dimensions)
```
Creates a tensor filled with random values from uniform distribution.

#### randn
```cpp
static Tensor<T> randn(Shape dimensions, T min, T max)
```
Creates a tensor filled with random values from a uniform distribution in the range [min, max].

**Note:** Despite the name "randn" (which typically suggests normal distribution), this function actually generates uniformly distributed random values.

**Example:**
```cpp
auto zeros = Tensor<float>::zeros(Shape{3, 3});
auto ones = Tensor<float>::ones(Shape{2, 2});
auto random = Tensor<float>::rand(Shape{5, 5});
auto randn = Tensor<float>::randn(Shape{10, 10}, -1.0f, 1.0f);
```

**Operators:**

#### Arithmetic Operations
```cpp
Tensor operator+(const Tensor& other) const
Tensor operator-(const Tensor& other) const
Tensor operator*(const Tensor& other) const  // Element-wise multiplication
Tensor operator/(const Tensor& other) const
```

#### Scalar Operations
```cpp
Tensor operator+(T scalar) const
Tensor operator-(T scalar) const
Tensor operator*(T scalar) const
Tensor operator/(T scalar) const
```

#### In-place Operations
```cpp
Tensor& operator+=(const Tensor& other)
Tensor& operator-=(const Tensor& other)
Tensor& operator*=(const Tensor& other)
Tensor& operator/=(const Tensor& other)
```

**Methods:**

#### reshape
```cpp
void reshape(const Shape& newShape)
void reshape(const std::vector<int>& dims)
```
Reshapes the tensor to new dimensions. Total number of elements must remain the same.

**Example:**
```cpp
Tensor<float> t(Shape{2, 3, 4});
t.reshape(Shape{6, 4});  // Valid: 2*3*4 = 6*4 = 24
```

#### slice
```cpp
Tensor<T> slice(int dim, int index) const
```
Extracts a slice along a specific dimension.

**Parameters:**
- `dim` - Dimension to slice along
- `index` - Index to extract

**Example:**
```cpp
Tensor<float> t(Shape{3, 4, 5});
auto slice = t.slice(0, 1);  // Returns tensor of shape {4, 5}
```

#### apply
```cpp
Tensor& apply(std::function<T(T)> func)
```
Applies a function element-wise to the tensor.

**Example:**
```cpp
tensor.apply([](float x) { return x * 2.0f; });
```

#### toDetailedString / toDataString
```cpp
std::string toDetailedString() const
std::string toDataString() const
```
Converts tensor to string representation for debugging.

---

### Layer

Abstract base class for all neural network layers.

**Template Parameters:**
- `T` - Data type (default: `float`)

**Virtual Methods:**

```cpp
virtual Tensor<T> forward(const Tensor<T>& input) = 0
virtual Tensor<T> backward(const Tensor<T>& gradOutput) = 0
virtual void updateWeights(Optimizer<T>& optimizer)
virtual void setTrainingMode(bool mode)
```

All layer implementations must override `forward()` and `backward()` methods.

---

### Loss

Abstract base class for loss functions.

**Template Parameters:**
- `T` - Data type (default: `float`)

**Virtual Methods:**

```cpp
virtual Tensor<T> compute(const Tensor<T>& prediction, const Tensor<T>& target) = 0
virtual Tensor<T> gradient(const Tensor<T>& prediction, const Tensor<T>& target) = 0
```

---

### Optimizer

Abstract base class for optimization algorithms.

**Template Parameters:**
- `T` - Data type (default: `float`)

**Virtual Methods:**

```cpp
virtual void optimize(
    const std::vector<std::reference_wrapper<Tensor<T>>>& weights,
    const std::vector<std::reference_wrapper<Tensor<T>>>& gradients,
    T learningRateOverride = T(-1.0)) = 0
```

---

## Layers

### FullyConnectedLayer

Dense/fully-connected layer that connects all inputs to all outputs.

**Constructor:**
```cpp
FullyConnectedLayer(int inputSize, int outputSize)
```

**Parameters:**
- `inputSize` - Number of input features
- `outputSize` - Number of output features

**Input Shape:**
- 1D: `(input_size)` or 2D: `(batch_size, input_size)`

**Output Shape:**
- 1D: `(output_size)` or 2D: `(batch_size, output_size)`

**Example:**
```cpp
// Maps 784 input features to 128 output features
FullyConnectedLayer layer(784, 128);
```

**Notes:**
- Weights are initialized randomly using Xavier/He initialization
- Biases are initialized to zero

---

### Conv2DLayer

2D convolutional layer for image processing.

**Constructor:**
```cpp
Conv2DLayer(int inChannels, int outChannels, int kernelSize, 
            int stride = 1, int padding = 0)
```

**Parameters:**
- `inChannels` - Number of input channels
- `outChannels` - Number of output feature maps
- `kernelSize` - Size of the convolution kernel (square)
- `stride` - Stride of convolution (default: 1)
- `padding` - Zero padding added to input (default: 0)

**Input Shape:**
- `(batch_size, channels, height, width)` or `(channels, height, width)`

**Output Shape:**
- `(batch_size, out_channels, out_height, out_width)` or `(out_channels, out_height, out_width)`

Where:
- `out_height = (height + 2*padding - kernelSize) / stride + 1`
- `out_width = (width + 2*padding - kernelSize) / stride + 1`

**Example:**
```cpp
// 1 input channel (grayscale), 32 output channels, 3x3 kernel
Conv2DLayer conv(1, 32, 3);
```

---

### ActivationLayer

Wrapper layer for applying activation functions.

**Constructor:**
```cpp
ActivationLayer(Activation<T> activation)
```

**Parameters:**
- `activation` - Activation function object (ReLU, Sigmoid, Tanh, Softmax, etc.)

**Example:**
```cpp
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(ActivationLayer(Sigmoid()));
model.addLayer(ActivationLayer(Softmax()));
```

---

### FlattenLayer

Flattens multi-dimensional input into 1D or 2D tensor.

**Constructor:**
```cpp
FlattenLayer()
```

**Input Shape:**
- Any multi-dimensional tensor

**Output Shape:**
- If input is 3D `(batch, h, w)`: output is 2D `(batch, h*w)`
- If input is 4D `(batch, c, h, w)`: output is 2D `(batch, c*h*w)`

**Example:**
```cpp
// Typical use after convolutional layers before fully connected
model.addLayer(Conv2DLayer(32, 64, 3));
model.addLayer(FlattenLayer());
model.addLayer(FullyConnectedLayer(flattened_size, 128));
```

---

## Regularization

### DropoutLayer

Randomly sets a fraction of input units to 0 during training to prevent overfitting.

**Constructor:**
```cpp
DropoutLayer(float dropoutRate)
```

**Parameters:**
- `dropoutRate` - Fraction of units to drop (0.0 to 1.0)

**Example:**
```cpp
// Drop 25% of units during training
model.addLayer(DropoutLayer(0.25f));
```

**Notes:**
- Only active during training mode
- Automatically disabled during evaluation
- Uses inverted dropout (scales remaining activations by 1/(1-p))

---

### BatchNormalizationLayer

Normalizes the activations of the previous layer at each batch.

**Constructor:**
```cpp
BatchNormalizationLayer(int numFeatures, float epsilon = 1e-5f, float momentum = 0.1f)
```

**Parameters:**
- `numFeatures` - Number of features/channels to normalize
- `epsilon` - Small constant for numerical stability (default: 1e-5)
- `momentum` - Momentum for running mean/variance (default: 0.1)

**Example:**
```cpp
// Normalize 128 features
model.addLayer(FullyConnectedLayer(256, 128));
model.addLayer(BatchNormalizationLayer(128));
model.addLayer(ActivationLayer(ReLU()));
```

**Notes:**
- Maintains running statistics (mean and variance) for inference
- During training: normalizes using batch statistics
- During evaluation: normalizes using running statistics

---

### MaxPooling2DLayer

Downsamples input by taking maximum value in each pooling window.

**Constructor:**
```cpp
MaxPooling2DLayer(int poolSize, int stride)
```

**Parameters:**
- `poolSize` - Size of the pooling window
- `stride` - Stride of pooling operation

**Input Shape:**
- `(batch_size, channels, height, width)`

**Output Shape:**
- `(batch_size, channels, height/stride, width/stride)`

**Example:**
```cpp
// 2x2 max pooling with stride 2 (halves spatial dimensions)
model.addLayer(MaxPooling2DLayer(2, 2));
```

---

## Activation Functions

### ReLU

Rectified Linear Unit activation: `f(x) = max(0, x)`

**Example:**
```cpp
model.addLayer(ActivationLayer(ReLU()));
```

**Properties:**
- Non-linear
- Sparse activation
- Can suffer from dying ReLU problem
- Most commonly used activation in hidden layers

---

### Sigmoid

Sigmoid activation: `f(x) = 1 / (1 + exp(-x))`

**Example:**
```cpp
model.addLayer(ActivationLayer(Sigmoid()));
```

**Properties:**
- Output range: (0, 1)
- Smooth gradient
- Can suffer from vanishing gradients
- Commonly used for binary classification output

---

### Tanh

Hyperbolic tangent activation: `f(x) = tanh(x)`

**Example:**
```cpp
model.addLayer(ActivationLayer(Tanh()));
```

**Properties:**
- Output range: (-1, 1)
- Zero-centered (unlike sigmoid)
- Can suffer from vanishing gradients

---

### Softmax

Softmax activation: `f(x_i) = exp(x_i) / sum(exp(x_j))`

**Example:**
```cpp
model.addLayer(ActivationLayer(Softmax()));
```

**Properties:**
- Output range: (0, 1)
- Outputs sum to 1 (can be interpreted as probabilities)
- Typically used in the output layer for multi-class classification

---

### LeakyReLU

Leaky ReLU activation: `f(x) = max(alpha*x, x)` where alpha is a small constant

**Example:**
```cpp
model.addLayer(ActivationLayer(LeakyReLU()));
```

**Properties:**
- Addresses dying ReLU problem
- Allows small negative values
- Default alpha typically 0.01

---

### Swish

Swish activation: `f(x) = x * sigmoid(x)`

**Example:**
```cpp
model.addLayer(ActivationLayer(Swish()));
```

**Properties:**
- Self-gated activation
- Smooth and non-monotonic
- Can outperform ReLU in deep networks

---

### Mish

Mish activation: `f(x) = x * tanh(softplus(x))`

**Example:**
```cpp
model.addLayer(ActivationLayer(Mish()));
```

**Properties:**
- Smooth and non-monotonic
- Self-regularizing
- Can improve accuracy in some tasks

---

## Optimizers

### AdamOptimizer

Adaptive Moment Estimation optimizer combining momentum and RMSprop.

**Constructor:**
```cpp
AdamOptimizer(const AdamOptions<T>& options = {})
```

**AdamOptions Structure:**
```cpp
struct AdamOptions {
    T learningRate = 0.001;   // Learning rate
    T beta1 = 0.9;            // Exponential decay rate for first moment
    T beta2 = 0.999;          // Exponential decay rate for second moment
    T epsilon = 1e-8;         // Small constant for numerical stability
    T l1Strength = 0;         // L1 regularization strength
    T l2Strength = 0;         // L2 regularization strength
    T decay = 0;              // Learning rate decay
    int batchSize = 1;        // Batch size for gradient averaging
};
```

**Example:**
```cpp
AdamOptions options;
options.learningRate = 0.001f;
options.beta1 = 0.9f;
options.beta2 = 0.999f;
options.l2Strength = 0.0001f;  // Add L2 regularization

model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer(options));
```

**Properties:**
- Adaptive learning rates for each parameter
- Works well with sparse gradients
- Requires minimal tuning
- Default choice for most problems

---

### SGDOptimizer

Stochastic Gradient Descent optimizer with optional momentum.

**Constructor:**
```cpp
SGDOptimizer(T learningRate = 0.01, T momentum = 0.0)
```

**Parameters:**
- `learningRate` - Learning rate for weight updates
- `momentum` - Momentum factor (0.0 to 1.0)

**Example:**
```cpp
model.compile(MSELoss(), SGDOptimizer(0.01f, 0.9f));
```

**Properties:**
- Simple and stable
- Momentum helps accelerate convergence
- May require learning rate scheduling

---

### RMSPropOptimizer

Root Mean Square Propagation optimizer with adaptive learning rates.

**Constructor:**
```cpp
RMSPropOptimizer(T learningRate = 0.001, T decay = 0.9, T epsilon = 1e-8)
```

**Parameters:**
- `learningRate` - Learning rate
- `decay` - Decay rate for moving average of squared gradients
- `epsilon` - Small constant for numerical stability

**Example:**
```cpp
model.compile(MSELoss(), RMSPropOptimizer(0.001f, 0.9f, 1e-8f));
```

**Properties:**
- Adapts learning rate for each parameter
- Works well for recurrent networks
- Good for non-stationary objectives

---

## Loss Functions

### MSELoss

Mean Squared Error loss for regression tasks.

**Formula:** `L = mean((prediction - target)^2)`

**Example:**
```cpp
model.compile(MSELoss(), AdamOptimizer());
```

**Use Cases:**
- Regression problems
- Continuous value prediction
- When outliers should be heavily penalized

---

### CategoricalCrossEntropyLoss

Categorical cross-entropy loss for multi-class classification.

**Formula:** `L = -sum(target * log(prediction))`

**Example:**
```cpp
model.compile(CategoricalCrossEntropyLoss(), AdamOptimizer());
```

**Use Cases:**
- Multi-class classification
- When outputs are one-hot encoded
- Typically used with Softmax activation in output layer

**Notes:**
- Expects one-hot encoded targets
- Prediction should be probability distribution (sum to 1)

---

## Utilities

### Shape

Represents the dimensions of a tensor.

**Constructor:**
```cpp
Shape(std::initializer_list<int> dims)
Shape(const std::vector<int>& dims)
```

**Example:**
```cpp
Shape shape1{28, 28};           // 2D shape
Shape shape2{32, 3, 224, 224};  // 4D shape (batch, channels, height, width)
```

**Methods:**
- `int operator[](size_t index) const` - Access dimension at index
- `size_t rank() const` - Number of dimensions
- `size_t size() const` - Total number of elements

---

### MNISTLoader

Utility class for loading MNIST dataset.

**Constructor:**
```cpp
MNISTLoader(const std::string& imagesPath, 
            const std::string& labelsPath,
            int batchSize = 32,
            int maxSamples = -1)
```

**Parameters:**
- `imagesPath` - Path to MNIST images file (IDX3 format)
- `labelsPath` - Path to MNIST labels file (IDX1 format)
- `batchSize` - Batch size for loading data
- `maxSamples` - Maximum samples to load (-1 for all)

**Methods:**

```cpp
std::pair<std::vector<Tensor<T>>, std::vector<Tensor<T>>> loadData()
std::string toAsciiArt(const Tensor<T>& image)
```

**Example:**
```cpp
MNISTLoader loader("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 64, 10000);
auto [inputs, targets] = loader.loadData();

// Display image as ASCII art
std::cout << loader.toAsciiArt(inputs[0]) << std::endl;
```

---

### SampleGenerator

Utility for generating synthetic datasets for testing.

**Functions:**

```cpp
std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> 
    generateLinearDataset(int num_samples, float noise = 1.0)
```

Generates linear regression dataset: `y = 2.0 * x + 3.0 + gaussian_noise`

The slope (2.0) and intercept (3.0) are hardcoded. The noise parameter controls the standard deviation of Gaussian (normal) noise added to the targets.

**Parameters:**
- `num_samples` - Total number of samples to generate in the dataset
- `noise` - Standard deviation of Gaussian noise added to targets (default: 1.0)

**Example:**
```cpp
auto [inputs, targets] = generateLinearDataset(1000);  // Default noise
auto [inputs, targets] = generateLinearDataset(1000, 0.5f);  // Less noise
```

---

## Type Aliases

SmartDNN supports template type parameters for flexibility:

```cpp
// Float precision (default)
SmartDNN<float> model;
Tensor<float> tensor;

// Double precision
SmartDNN<double> model;
Tensor<double> tensor;
```

**Recommendation:** Use `float` for most applications unless you specifically need higher precision, as it's faster and uses less memory.

---

## Performance Tips

1. **Batch Processing**: Process multiple samples together for better performance
2. **Memory Layout**: SmartDNN uses row-major layout for tensors
3. **Template Specialization**: The library uses templates for performance optimization
4. **Parallel Operations**: Many operations are parallelized internally
5. **View Operations**: Use slice/broadcast views to avoid data copying when possible

---

## Error Handling

SmartDNN uses C++ exceptions for error handling:

- `std::invalid_argument` - Invalid parameters or arguments
- `std::runtime_error` - Runtime errors during computation
- `std::out_of_range` - Index out of bounds

Always wrap model operations in try-catch blocks for production code:

```cpp
try {
    model.train(inputs, targets, epochs);
} catch (const std::exception& e) {
    std::cerr << "Training failed: " << e.what() << std::endl;
}
```

---

## Version Information

This documentation corresponds to SmartDNN version 1.0.0.

For the latest updates and changes, see the [CHANGELOG](../CHANGELOG.md) (if available).

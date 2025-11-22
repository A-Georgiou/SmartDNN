# SmartDNN Performance Guide

This guide provides best practices, optimization techniques, and benchmarking information for maximizing performance with SmartDNN.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Compilation Optimizations](#compilation-optimizations)
- [Model Architecture Optimizations](#model-architecture-optimizations)
- [Training Optimizations](#training-optimizations)
- [Memory Management](#memory-management)
- [Benchmarking](#benchmarking)
- [Profiling](#profiling)
- [Platform-Specific Optimizations](#platform-specific-optimizations)
- [Common Performance Pitfalls](#common-performance-pitfalls)

---

## Performance Overview

SmartDNN leverages modern C++ features and compiler optimizations to achieve high performance:

### Performance Highlights

**Linear Regression (1000 samples, 1000 epochs):**
- Non-templated runtime: ~17,680ms
- Templated runtime: ~8,325ms
- **Improvement: ~53%**

**MNIST Classification (1000 samples, batch size 64, 1000 epochs):**
- Non-templated runtime: ~83 minutes per epoch
- Templated runtime: ~10,969ms per epoch
- **Improvement: ~99.8%**

### Key Performance Features

1. **Template-based design**: Zero-cost abstractions
2. **Move semantics**: Reduced copying
3. **Parallel operations**: OpenMP support
4. **Cache-friendly layout**: Contiguous memory
5. **Compiler optimizations**: Aggressive inlining and optimization flags

---

## Compilation Optimizations

### Build Types

#### Release Build (Recommended for Production)

```bash
# Configure for release build
cmake -DCMAKE_BUILD_TYPE=Release .
make

# Alternative: specify optimization flags directly
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" .
make
```

**Optimization Flags (GCC/Clang):**
- `-O3`: Aggressive optimizations
- `-march=native`: CPU-specific optimizations
- `-flto`: Link-time optimization (optional)
- `-ffast-math`: Fast floating-point math (use with caution)

**Optimization Flags (MSVC):**
- `/O2`: Maximum optimizations
- `/arch:AVX2`: AVX2 instructions (if supported)

#### Debug Build (For Development)

```bash
cmake -DCMAKE_BUILD_TYPE=Debug .
make
```

**Debug Flags:**
- `-g`: Debug symbols
- `-O0`: No optimization
- `ENABLE_LOGGING`: Logging enabled

### Compiler Selection

Different compilers may produce different performance:

```bash
# GCC
export CC=gcc
export CXX=g++
cmake .

# Clang (often better optimizations)
export CC=clang
export CXX=clang++
cmake .

# Compare performance with both
```

### Link-Time Optimization (LTO)

Enable LTO for additional performance:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON .
make
```

---

## Model Architecture Optimizations

### 1. Layer Selection

Choose appropriate layers for your task:

```cpp
// For image data: Use Conv2D instead of FullyConnected
// Good (efficient for images):
model.addLayer(Conv2DLayer(1, 32, 3));
model.addLayer(MaxPooling2DLayer(2, 2));

// Bad (inefficient for images):
model.addLayer(FullyConnectedLayer(784, 10000));  // Too many parameters
```

### 2. Network Depth vs. Width

**Trade-offs:**
- **Deeper networks**: Better feature learning, slower training
- **Wider networks**: More parameters, better capacity, more memory

```cpp
// Deeper (better for complex patterns)
model.addLayer(FullyConnectedLayer(128, 64));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(FullyConnectedLayer(64, 32));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(FullyConnectedLayer(32, 10));

// Wider (faster, more memory)
model.addLayer(FullyConnectedLayer(128, 256));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(FullyConnectedLayer(256, 10));
```

### 3. Activation Functions

Different activations have different computational costs:

```cpp
// ReLU: Fastest (simple max operation)
model.addLayer(ActivationLayer(ReLU()));

// Sigmoid: Slower (exponential computation)
model.addLayer(ActivationLayer(Sigmoid()));

// Tanh: Slower (hyperbolic tangent)
model.addLayer(ActivationLayer(Tanh()));

// Softmax: Slowest (exponentials + normalization)
model.addLayer(ActivationLayer(Softmax()));  // Only use for output layer
```

**Recommendation:** Use ReLU for hidden layers, Softmax only for output.

### 4. Batch Normalization Placement

```cpp
// Efficient placement: After linear transformation, before activation
model.addLayer(FullyConnectedLayer(128, 64));
model.addLayer(BatchNormalizationLayer(64));  // Normalize here
model.addLayer(ActivationLayer(ReLU()));

// Less efficient: After activation
model.addLayer(FullyConnectedLayer(128, 64));
model.addLayer(ActivationLayer(ReLU()));
model.addLayer(BatchNormalizationLayer(64));  // Avoid this
```

### 5. Convolution Parameters

Optimize convolution layer parameters:

```cpp
// Larger stride = faster but less information
model.addLayer(Conv2DLayer(32, 64, 3, 2));  // stride=2 (faster)
model.addLayer(Conv2DLayer(32, 64, 3, 1));  // stride=1 (slower, more accurate)

// Use pooling instead of large strides for downsampling
model.addLayer(Conv2DLayer(32, 64, 3, 1));
model.addLayer(MaxPooling2DLayer(2, 2));  // Separate downsampling
```

---

## Training Optimizations

### 1. Batch Size

Larger batches = better throughput:

```cpp
// Small batch (slower, more updates)
constexpr int BATCH_SIZE = 8;

// Large batch (faster, fewer updates)
constexpr int BATCH_SIZE = 128;

// Find optimal batch size for your hardware
// Typical values: 32, 64, 128, 256
```

**Trade-offs:**
- **Larger batches**: Faster per-sample, less frequent updates, more memory
- **Smaller batches**: Slower per-sample, more frequent updates, less memory

**Finding optimal batch size:**

```cpp
// Test different batch sizes
for (int batchSize : {8, 16, 32, 64, 128, 256}) {
    auto start = std::chrono::high_resolution_clock::now();
    
    MNISTLoader loader(imagePath, labelPath, batchSize, 1000);
    auto [inputs, targets] = loader.loadData();
    model.train(inputs, targets, 1);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Batch size " << batchSize << ": " 
              << duration.count() << "ms" << std::endl;
}
```

### 2. Optimizer Selection

Different optimizers have different performance:

```cpp
// Adam: Slower but usually better convergence
AdamOptions adamOpt;
adamOpt.learningRate = 0.001f;
model.compile(loss, AdamOptimizer(adamOpt));

// SGD: Faster but may need tuning
model.compile(loss, SGDOptimizer(0.01f, 0.9f));

// RMSProp: Middle ground
model.compile(loss, RMSPropOptimizer(0.001f));
```

**Performance ranking (fastest to slowest):**
1. SGD (simplest)
2. RMSProp
3. Adam (most computations, but often best results)

### 3. Learning Rate

Proper learning rate can speed up convergence:

```cpp
// Too small: slow convergence
options.learningRate = 0.00001f;  // Many epochs needed

// Too large: instability, divergence
options.learningRate = 1.0f;  // May not converge

// Good starting point
options.learningRate = 0.001f;  // Usually works well

// Use learning rate scheduling
// Start high, decrease over time
options.learningRate = 0.01f;   // First 50 epochs
model.train(inputs, targets, 50);

// Reduce learning rate and recompile
options.learningRate = 0.001f;  // Next 50 epochs
model.compile(MSELoss(), AdamOptimizer(options));  // Recompile with new settings
model.train(inputs, targets, 50);
```

### 4. Early Stopping

Don't train longer than necessary:

```cpp
// Monitor validation loss and stop early
float bestLoss = std::numeric_limits<float>::max();
int patienceCounter = 0;
const int patience = 10;

for (int epoch = 0; epoch < maxEpochs; epoch++) {
    model.train(trainInputs, trainTargets, 1);
    
    model.evalMode();
    float valLoss = computeLoss(valInputs, valTargets);
    model.trainingMode();
    
    if (valLoss < bestLoss) {
        bestLoss = valLoss;
        patienceCounter = 0;
        model.saveModel("best_model.bin");
    } else {
        patienceCounter++;
        if (patienceCounter >= patience) {
            std::cout << "Early stopping at epoch " << epoch << std::endl;
            break;
        }
    }
}
```

### 5. Data Loading

Minimize data loading overhead:

```cpp
// Bad: Load data every epoch
for (int epoch = 0; epoch < epochs; epoch++) {
    auto [inputs, targets] = loader.loadData();  // Slow!
    model.train(inputs, targets, 1);
}

// Good: Load once, reuse
auto [inputs, targets] = loader.loadData();
model.train(inputs, targets, epochs);
```

---

## Memory Management

### 1. Tensor Reuse

Reuse tensors to avoid allocations:

```cpp
// Bad: Create new tensors in loop
for (int i = 0; i < iterations; i++) {
    Tensor<float> temp(Shape{1000, 1000});  // Allocates every iteration
    // Use temp
}

// Good: Allocate once, reuse
Tensor<float> temp(Shape{1000, 1000});
for (int i = 0; i < iterations; i++) {
    // Reuse temp
    temp = computeSomething();
}
```

### 2. Move Semantics

Use move operations to avoid copying:

```cpp
// Bad: Copy
Tensor<float> a = createLargeTensor();
Tensor<float> b = a;  // Copies data

// Good: Move
Tensor<float> a = createLargeTensor();
Tensor<float> b = std::move(a);  // Moves data, no copy

// Also good: In-place operations
tensor *= 2.0f;  // In-place, no allocation
```

### 3. Memory Pools (Future Enhancement)

```cpp
// Current: Individual allocations
// Future: Memory pool for tensors
```

### 4. Gradient Accumulation

For large models with limited memory:

```cpp
// Accumulate gradients over multiple batches
constexpr int virtualBatchSize = 256;
constexpr int actualBatchSize = 32;
constexpr int accumSteps = virtualBatchSize / actualBatchSize;

for (int step = 0; step < accumSteps; step++) {
    // Forward and backward on small batch
    // Accumulate gradients
}
// Update weights once with accumulated gradients
```

---

## Benchmarking

### Basic Benchmarking

```cpp
#include <chrono>

int main() {
    // Setup
    SmartDNN<float> model;
    // ... configure model ...
    
    auto [inputs, targets] = loadData();
    
    // Benchmark training
    auto start = std::chrono::high_resolution_clock::now();
    
    model.train(inputs, targets, 100);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << "ms" << std::endl;
    std::cout << "Time per epoch: " << duration.count() / 100.0 << "ms" << std::endl;
    
    return 0;
}
```

### Inference Benchmarking

```cpp
// Benchmark single inference
auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i < 1000; i++) {
    auto prediction = model.predict(testInput);
}

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

std::cout << "Average inference time: " 
          << duration.count() / 1000.0 << "µs" << std::endl;
```

### Throughput Measurement

```cpp
// Measure samples per second
int totalSamples = inputs.size() * epochs;
auto start = std::chrono::high_resolution_clock::now();

model.train(inputs, targets, epochs);

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

std::cout << "Throughput: " 
          << totalSamples / static_cast<float>(duration.count())
          << " samples/second" << std::endl;
```

---

## Profiling

### Using GNU gprof

```bash
# Compile with profiling
g++ -pg -O2 src/main.cpp -o SmartDNN

# Run program
./SmartDNN

# Generate profile
gprof SmartDNN gmon.out > analysis.txt

# View analysis
less analysis.txt
```

### Using Valgrind (Memory Profiling)

```bash
# Install valgrind
sudo apt-get install valgrind

# Run with callgrind
valgrind --tool=callgrind ./SmartDNN

# Visualize with kcachegrind
kcachegrind callgrind.out.*
```

### Using perf (Linux)

```bash
# Record performance data
perf record -g ./SmartDNN

# Analyze
perf report
```

---

## Platform-Specific Optimizations

### Linux

```bash
# Enable CPU performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check NUMA topology
numactl --hardware

# Bind to specific CPU cores
taskset -c 0-7 ./SmartDNN
```

### Windows

```cpp
// Set high priority (Windows API)
#ifdef _WIN32
#include <windows.h>
SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
```

### macOS

```bash
# Enable performance mode
sudo pmset -a cpufreq max
```

### CPU-Specific Flags

```bash
# Intel CPUs
g++ -O3 -march=native -mtune=native

# AMD CPUs
g++ -O3 -march=znver2  # For Zen 2

# ARM CPUs
g++ -O3 -march=armv8-a+simd
```

---

## Common Performance Pitfalls

### 1. Debug Builds in Production

```bash
# DON'T: Use debug builds for performance testing
cmake -DCMAKE_BUILD_TYPE=Debug .

# DO: Use release builds
cmake -DCMAKE_BUILD_TYPE=Release .
```

### 2. Excessive Logging

```cpp
// DON'T: Log in tight loops
for (int i = 0; i < 1000000; i++) {
    std::cout << "Processing " << i << std::endl;  // Very slow!
}

// DO: Log sparingly
if (i % 10000 == 0) {
    std::cout << "Processing " << i << std::endl;
}
```

### 3. Unnecessary Copies

```cpp
// DON'T: Pass large objects by value
void process(Tensor<float> data) {  // Copies!
    // ...
}

// DO: Pass by const reference
void process(const Tensor<float>& data) {
    // ...
}
```

### 4. Small Batch Sizes

```cpp
// DON'T: Use very small batches (unless necessary)
constexpr int BATCH_SIZE = 1;  // Very inefficient

// DO: Use reasonable batch sizes
constexpr int BATCH_SIZE = 64;  // Much better
```

### 5. Inappropriate Data Types

```cpp
// DON'T: Use double when float suffices
SmartDNN<double> model;  // Slower, more memory

// DO: Use float for most applications
SmartDNN<float> model;  // Faster, less memory
```

### 6. Loading Data Multiple Times

```cpp
// DON'T: Reload data unnecessarily
for (int epoch = 0; epoch < 100; epoch++) {
    auto data = loadData();  // Slow!
    model.train(data.first, data.second, 1);
}

// DO: Load once
auto [inputs, targets] = loadData();
model.train(inputs, targets, 100);
```

---

## Performance Checklist

Before deploying or benchmarking:

- [ ] Using Release build mode
- [ ] Optimization flags enabled (-O3, -march=native)
- [ ] Appropriate batch size selected
- [ ] Using fast activation functions (ReLU) where possible
- [ ] Data loaded efficiently
- [ ] No unnecessary logging
- [ ] Proper data types (float vs. double)
- [ ] Move semantics utilized
- [ ] Tensors reused when possible
- [ ] Profiled code to identify bottlenecks

---

## Performance Tuning Workflow

1. **Baseline**: Measure current performance
2. **Profile**: Identify bottlenecks
3. **Optimize**: Apply targeted optimizations
4. **Measure**: Verify improvements
5. **Repeat**: Continue optimizing if needed

---

## Expected Performance

### Reference Hardware: Intel i7-9700K, 16GB RAM

| Task | Input Size | Batch | Epochs | Time |
|------|-----------|-------|--------|------|
| Linear Regression | 1000 samples | 100 | 1000 | ~8.3s |
| MNIST CNN | 1000 samples | 64 | 1000 | ~11s/epoch |
| XOR Problem | 400 samples | 4 | 200 | <1s |

Your performance may vary based on:
- CPU model and speed
- Available RAM
- Compiler and optimization flags
- Model architecture
- Data size

---

## Future Optimizations

Planned performance improvements:

1. **GPU Support**: CUDA backend for tensor operations
2. **SIMD Optimizations**: Explicit vectorization
3. **Mixed Precision**: FP16 training
4. **Distributed Training**: Multi-GPU/multi-node
5. **Graph Optimization**: Computation graph fusion
6. **Custom Allocators**: Memory pool optimization

---

## Conclusion

Performance optimization is an iterative process. Start with the basics (Release build, good batch size), profile to find bottlenecks, and apply targeted optimizations. Always measure the impact of your changes!

For more information:
- [API Reference](API_REFERENCE.md)
- [Tutorials](TUTORIALS.md)
- [Architecture Guide](ARCHITECTURE.md)

# SmartDNN Test Suite

This directory contains comprehensive unit tests for the SmartDNN deep learning library.

## Test Coverage

The test suite includes **139 tests** covering all major components:

### Tensor Operations (46 tests)
- Basic tensor initialization and operations
- Element-wise operations (addition, subtraction, multiplication, division)
- Scalar operations
- Copy and move semantics
- Advanced operations (transpose, reshape, matrix multiplication, etc.)
- Broadcasting and slicing

### Activations (7 tests)
- ReLU, Leaky ReLU
- Sigmoid, Tanh
- Softmax
- Swish, Mish
- Forward and backward passes

### Layers (29 tests)
- **Fully Connected Layer**: Forward/backward passes, batched processing
- **Conv2D Layer**: Convolution operations, gradient computation
- **Flatten Layer**: Dimension flattening, shape preservation
- **Activation Layer**: Wrapper functionality for activations
- **MaxPooling2D Layer**: Max pooling with different strides and channels

### Regularization (6 tests)
- Batch Normalization (2D and 4D inputs)
- Dropout (training and inference modes)

### Loss Functions (12 tests)
- **MSE Loss**: Mean squared error computation and gradients
- **Categorical Cross Entropy**: Multi-class classification loss

### Optimizers (24 tests)
- **Adam Optimizer**: Adaptive learning rates, momentum, weight decay
- **SGD Optimizer**: Stochastic gradient descent with momentum and Nesterov
- **RMSProp Optimizer**: Root mean square propagation

## Running Tests

### Quick Start

From the repository root, run:

```bash
./run_tests.sh
```

This script will:
1. Configure the project with CMake
2. Build all tests
3. Run the complete test suite
4. Display results with colored output

### Clean Build

To force a clean rebuild:

```bash
./run_tests.sh clean
```

### Manual Testing

If you prefer to run tests manually:

```bash
cd tests
cmake .
make
./RunTests
```

### Run Specific Tests

To run specific test suites, use Google Test filters:

```bash
cd tests
./RunTests --gtest_filter="TensorOperatorTest.*"           # Run only tensor operator tests
./RunTests --gtest_filter="MSELossTest.*"                  # Run only MSE loss tests
./RunTests --gtest_filter="AdamOptimizerTest.*"            # Run only Adam optimizer tests
```

## Test Organization

```
tests/
├── activations/          # Activation function tests
│   └── test_activations.cpp
├── layers/               # Layer tests
│   ├── test_additional_layers.cpp
│   ├── test_conv_2d.cpp
│   └── test_fully_connected.cpp
├── loss/                 # Loss function tests
│   └── test_loss_functions.cpp
├── optimizers/           # Optimizer tests
│   ├── test_adam_optimizer.cpp
│   ├── test_rmsprop_optimizer.cpp
│   └── test_sgd_optimizer.cpp
├── tensor/               # Tensor operation tests
│   ├── test_advanced_tensor_operations.cpp
│   ├── test_layers.cpp
│   └── test_tensor.cpp
├── utils/                # Test utilities
│   └── tensor_helpers.hpp
├── main.cpp              # Test entry point
└── CMakeLists.txt        # CMake configuration
```

## Requirements

- CMake 3.10 or higher
- C++17 compatible compiler
- Google Test (automatically fetched by CMake)

## Test Results

All 139 tests pass successfully:

```
[==========] 139 tests from 24 test suites ran. (58 ms total)
[  PASSED  ] 139 tests.
```

## Contributing

When adding new features to SmartDNN:

1. Add corresponding tests in the appropriate directory
2. Follow existing test patterns and naming conventions
3. Ensure all tests pass before submitting changes
4. Aim for comprehensive coverage of edge cases

## Test Utilities

The `utils/tensor_helpers.hpp` file provides helper functions for:
- Validating tensor shapes
- Comparing tensor data
- Checking tensor equality
- Common test assertions

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. The `run_tests.sh` script returns:
- Exit code 0: All tests passed
- Exit code 1: One or more tests failed

This makes it easy to integrate with automated testing systems.

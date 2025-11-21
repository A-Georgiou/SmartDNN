# Test Coverage Documentation

## Overview
This document describes the comprehensive test suite for the SmartDNN deep learning library. The test suite uses Google Test framework and covers core components including tensors, layers, activations, optimizers, and loss functions.

## Test Statistics
- **Total Tests**: 155
- **Test Suites**: 26
- **Test Execution Time**: ~60ms
- **Pass Rate**: 100%

## Test Organization

### Directory Structure
```
tests/
├── activations/          # Activation function tests
│   └── test_activations.cpp
├── layers/              # Layer tests
│   ├── test_conv_2d.cpp
│   ├── test_fully_connected.cpp
│   └── test_additional_layers.cpp
├── loss/                # Loss function tests
│   └── test_loss_functions.cpp
├── optimizers/          # Optimizer tests
│   ├── test_adam_optimizer.cpp
│   ├── test_rmsprop_optimizer.cpp
│   └── test_sgd_optimizer.cpp
├── tensor/              # Tensor operation tests
│   ├── test_tensor.cpp
│   ├── test_advanced_tensor_operations.cpp
│   └── test_layers.cpp
├── utils/               # Utility and helper tests
│   ├── test_shape_and_factory.cpp
│   └── tensor_helpers.hpp
└── main.cpp             # Test runner
```

## Component Test Coverage

### Activation Functions (14 tests)
All 7 activation functions are fully tested:
- **ReLU**: Forward and backward pass (2 tests)
- **LeakyReLU**: Forward and backward pass with negative slope (2 tests)
- **Sigmoid**: Forward and backward pass (2 tests)
- **Tanh**: Forward and backward pass (2 tests)
- **Softmax**: Forward and backward pass (2 tests)
- **Swish**: Forward and backward pass (2 tests)
- **Mish**: Forward and backward pass (2 tests)

### Loss Functions (12 tests)
#### MSELoss (6 tests)
- Compute with matching shapes
- Compute with perfect prediction
- Compute with reshapable target
- Compute with mismatched shapes (error case)
- Gradient computation
- Gradient with zero difference

#### CategoricalCrossEntropyLoss (6 tests)
- Compute basic loss
- Compute with perfect prediction
- Compute with mismatched shapes (error case)
- Gradient computation
- Gradient with mismatched shapes (error case)
- Numerical stability with extreme values

### Optimizers (19 tests)
#### SGDOptimizer (5 tests)
- Basic SGD without momentum
- SGD with momentum
- SGD with Nesterov momentum
- Weight decay
- Learning rate override

#### RMSPropOptimizer (6 tests)
- Basic RMSProp update
- Multiple iterations
- Learning rate decay
- Learning rate override
- Multiple weight tensors
- Size mismatch error

#### AdamOptimizer (8 tests)
- Basic Adam update
- Multiple iterations
- Learning rate override
- L1 regularization
- L2 regularization
- Learning rate decay
- Multiple weight tensors
- Size mismatch error

### Layers (31 tests)
#### FullyConnectedLayer (9 tests)
- Forward pass
- Backward pass
- Weight initialization
- Bias handling
- Gradient computation
- Multiple batch sizes
- Different layer sizes
- Weight updates
- Error handling

#### Conv2DLayer (8 tests)
- Forward pass basic
- Forward pass with padding
- Forward pass with stride
- Multiple channels
- Backward pass
- Weight gradients
- Bias gradients
- Error handling

#### FlattenLayer (5 tests)
- Forward pass with 2D input
- Forward pass with 3D input
- Forward pass with 4D input
- Backward pass
- Invalid input (1D) error

#### MaxPooling2DLayer (6 tests)
- Forward pass basic
- Forward with multiple channels
- Forward with stride 1
- Backward pass
- Invalid input rank error
- Backward without forward error

#### ActivationLayer (3 tests)
- Forward with ReLU
- Backward with ReLU
- Preserve input shape

### Regularization Layers (6 tests)
#### BatchNormalizationLayer (3 tests)
- Forward pass 2D
- Forward pass 4D
- Backward pass

#### DropoutLayer (3 tests)
- Forward pass training mode
- Forward pass inference mode
- Backward pass

### Tensor Operations (60 tests)
#### Basic Tensor Operations (22 tests)
- Tensor initialization (3 tests)
- Element-wise operators (8 tests)
- Scalar operators (8 tests)
- Copy and move semantics (3 tests)

#### Advanced Tensor Operations (35 tests)
- Apply functions
- Sum operations (across different axes)
- Reciprocal operations
- Mean operations
- Variance multiplication
- Transpose operations
- Reshape operations
- Dot product
- Matrix-vector multiplication
- Matrix-matrix multiplication
- Batched matrix multiplication

#### Regularization Layers (3 tests)
- Dropout layer (forward training, forward inference, backward)

### Shape and Utilities (16 tests)
#### Shape (12 tests)
- Construction from initializer list
- Construction from vector
- Copy constructor
- Move constructor
- Copy assignment
- Move assignment
- Equality and inequality
- ToString conversion
- Invalid dimensions validation
- Zero dimension handling
- Get dimensions

#### ShapeOperations (4 tests)
- Broadcast 2D shapes
- Broadcast with scalar
- Check broadcastability (positive case)
- Check broadcastability (negative case)

## Running Tests

### Build Tests
```bash
cd tests
cmake .
make -j$(nproc)
```

### Run All Tests
```bash
./RunTests
```

### Run Specific Test Suite
```bash
./RunTests --gtest_filter=MSELossTest.*
./RunTests --gtest_filter=AdamOptimizerTest.*
./RunTests --gtest_filter=FlattenLayerTest.*
```

### Run Specific Test
```bash
./RunTests --gtest_filter=MSELossTest.ComputeWithMatchingShapes
```

### Verbose Output
```bash
./RunTests --gtest_verbose=1
```

## Test Helpers

### Tensor Validation Helpers (tensor_helpers.hpp)
- `ValidateTensorShape`: Check tensor dimensions
- `ValidateTensorData`: Compare tensor data with expected values
- `ValidateRandomTensor`: Verify random values are in range
- `TensorEquals`: Compare two tensors for equality

## Writing New Tests

### Test File Template
```cpp
#ifndef TEST_YOUR_COMPONENT_CPP
#define TEST_YOUR_COMPONENT_CPP

#include <gtest/gtest.h>
#include "../../smart_dnn/YourComponent.hpp"
#include "../utils/tensor_helpers.hpp"

namespace smart_dnn {

TEST(YourComponentTest, BasicFunctionality) {
    // Arrange
    YourComponent component;
    Tensor<float> input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    // Act
    Tensor<float> output = component.process(input);
    
    // Assert
    EXPECT_EQ(output.getShape().rank(), 2);
    ValidateTensorData(output, expectedData);
}

} // namespace smart_dnn

#endif // TEST_YOUR_COMPONENT_CPP
```

### Best Practices
1. **Test both success and failure cases**
   - Valid inputs
   - Invalid inputs (shape mismatches, etc.)
   - Boundary conditions

2. **Use descriptive test names**
   - Pattern: `ComponentTest.DescriptiveActionAndExpectation`
   - Example: `MSELossTest.ComputeWithMatchingShapes`

3. **Keep tests focused**
   - One concept per test
   - Clear arrange-act-assert structure

4. **Use helper functions**
   - Leverage `ValidateTensorShape` and `ValidateTensorData`
   - Extract common setup to helper functions

5. **Test edge cases**
   - Empty tensors
   - Zero values
   - Extreme values (very large/small)
   - Shape mismatches

## Components Not Yet Tested

The following components are not covered by the current test suite:

### Advanced Features
- **TensorFactory**: Factory methods for tensor creation (has circular dependency issues)
- **SliceView**: Tensor slicing views
- **BroadcastView**: Broadcasting views
- **RandomEngine**: Random number generation utilities

### Utilities
- **Logger**: Debugging and logging utilities
- **Datasets**: MNIST loader and sample generators

### Integration
- **SmartDNN**: Main model class (integration testing)

These components could be added in future test iterations as they require more complex setup or integration testing approaches.

## Performance Considerations

The test suite is designed to be fast:
- All tests complete in ~60ms
- Parallel compilation supported (`make -j`)
- Small test datasets for quick execution
- No external dependencies beyond Google Test

## Contributing Tests

When contributing new tests:

1. **Follow existing patterns**
   - Use the same test structure as similar components
   - Maintain consistency in naming conventions

2. **Add to CMakeLists.txt**
   - Tests are automatically discovered via `file(GLOB_RECURSE TEST_SOURCES ...)`
   - Just place your test file in the appropriate directory

3. **Document test coverage**
   - Update this README with new test counts
   - Describe what's being tested

4. **Ensure all tests pass**
   - Run the full test suite before submitting
   - Fix any failing tests

## Continuous Integration

Tests are automatically built and run:
- Clean build from CMake configuration
- All 155 tests must pass
- No compiler warnings tolerated

## Test Maintenance

### Regular Updates Needed
- When adding new components, add corresponding tests
- When fixing bugs, add regression tests
- Keep test data small and focused
- Update documentation when test structure changes

### Code Coverage Goals
- Aim for >90% code coverage on core components
- 100% coverage on critical paths (forward/backward passes)
- Error handling should be tested for all edge cases

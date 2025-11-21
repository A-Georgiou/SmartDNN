# SmartDNN Testing

This directory contains the test suite for SmartDNN using Google Test framework.

## Running Tests

### Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler
- Make

### Building and Running Tests

```bash
cd tests
cmake .
make -j$(nproc)
./RunTests
```

### Running Specific Tests

You can run specific test suites or tests using GoogleTest filters:

```bash
# Run only Tensor tests
./RunTests --gtest_filter=Tensor*

# Run a specific test
./RunTests --gtest_filter=TensorInitialisationTest.ExpectDefaultShapeInitialisation

# Run all tests except specific ones
./RunTests --gtest_filter=-*ZeroDimensions*
```

### Test Output Options

```bash
# Verbose output
./RunTests --gtest_verbose

# List all available tests without running them
./RunTests --gtest_list_tests

# Generate XML output for CI/CD
./RunTests --gtest_output=xml:test-results.xml
```

## Test Structure

The test suite is organized into the following categories:

### Tensor Tests (`tensor/`)
- **test_tensor.cpp**: Basic tensor initialization, copy/move semantics, and operators
- **test_tensor_operations.cpp**: Tensor operations like matmul, transpose, etc.

### Activation Tests (`activation/`)
- **test_activations.cpp**: Tests for activation functions (ReLU, Sigmoid, Softmax, etc.)

### Utilities (`utils/`)
- **tensor_helpers.hpp**: Helper functions for validating tensor shape and data

## Adding New Tests

To add new tests:

1. Create a new test file in the appropriate subdirectory (e.g., `tests/layers/test_fully_connected.cpp`)
2. Include the GoogleTest headers and necessary SmartDNN headers
3. Write your tests using the `TEST()` macro
4. The test will be automatically included in the build (CMake uses `GLOB_RECURSE`)

Example:
```cpp
#include <gtest/gtest.h>
#include "../smart_dnn/YourClass.hpp"

TEST(YourTestSuite, YourTestCase) {
    // Your test code
    ASSERT_EQ(expected, actual);
}
```

## Continuous Integration

Tests are automatically run on every push and pull request via GitHub Actions. See `.github/workflows/tests.yml` for the CI configuration.

## Test Results

Current test status:
- **58/60 tests passing** (2 tests for zero-dimension edge cases are known to fail)
- Average test execution time: < 100ms

## Debugging Failed Tests

If a test fails:

1. Run the test with verbose output: `./RunTests --gtest_filter=FailedTest* --gtest_verbose`
2. Check the test output for assertion failures
3. Use a debugger (gdb) to step through the test:
   ```bash
   gdb ./RunTests
   (gdb) run --gtest_filter=FailedTest*
   ```

## Known Issues

- Two tests for zero-dimension tensors fail as the library doesn't support zero-sized dimensions
- These are pre-existing issues and not related to the test infrastructure

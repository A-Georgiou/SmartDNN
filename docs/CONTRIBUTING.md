# Contributing to SmartDNN

Thank you for your interest in contributing to SmartDNN! This guide will help you get started with contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Making Changes](#making-changes)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards other contributors

**Unacceptable behaviors include:**
- Harassment, trolling, or inflammatory comments
- Personal attacks or derogatory language
- Publishing others' private information
- Any conduct inappropriate in a professional setting

---

## Getting Started

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10 or higher
- Git
- Basic understanding of neural networks and C++

### First Contributions

Good first issues for new contributors:
- Documentation improvements
- Adding code examples
- Writing tests for existing features
- Fixing typos or formatting issues

Look for issues tagged with `good-first-issue` or `help-wanted`.

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/SmartDNN.git
cd SmartDNN

# Add upstream remote
git remote add upstream https://github.com/A-Georgiou/SmartDNN.git
```

### 2. Build the Project

```bash
# Configure with CMake
cmake .

# Build
make

# Verify build
./SmartDNN
```

### 3. Create a Branch

```bash
# Update your local repository
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch
git checkout -b feature/my-new-feature
```

### 4. Development Build

For development with debugging symbols:

```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug .
make

# This enables logging and debugging symbols
```

For optimized builds:

```bash
# Release build
cmake -DCMAKE_BUILD_TYPE=Release .
make
```

---

## Project Structure

```
SmartDNN/
├── smart_dnn/              # Main library source code
│   ├── SmartDNN.hpp        # Main model class
│   ├── Layer.hpp           # Base layer interface
│   ├── Loss.hpp            # Base loss interface
│   ├── Optimizer.hpp       # Base optimizer interface
│   ├── Layers/             # Layer implementations
│   ├── Activations/        # Activation functions
│   ├── Optimizers/         # Optimizer implementations
│   ├── Loss/               # Loss function implementations
│   ├── Tensor/             # Tensor and operations
│   ├── Shape/              # Shape utilities
│   ├── Regularisation/     # Regularization layers
│   ├── Datasets/           # Dataset loaders
│   └── Debugging/          # Debug utilities
├── examples/               # Example programs
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── src/                    # User application source (not tracked)
├── CMakeLists.txt          # Build configuration
└── README.md               # Project overview
```

### File Naming Conventions

- **Headers:** PascalCase with `.hpp` extension (e.g., `FullyConnectedLayer.hpp`)
- **Implementation:** Same name with `.impl.hpp` extension for template implementations
- **Source:** `.cpp` extension for non-template code
- **Tests:** `test_*.cpp` pattern

---

## Coding Standards

### C++ Style Guide

#### 1. Naming Conventions

```cpp
// Classes and Structs: PascalCase
class FullyConnectedLayer { };
struct AdamOptions { };

// Functions and Methods: camelCase
void updateWeights() { }
Tensor forward(const Tensor& input) { }

// Variables: camelCase
int numLayers = 0;
float learningRate = 0.01f;

// Constants: UPPER_SNAKE_CASE or camelCase
constexpr int MAX_ITERATIONS = 1000;
const float defaultLearningRate = 0.001f;

// Template Parameters: PascalCase
template <typename T, typename DeviceType>

// Namespaces: snake_case
namespace smart_dnn { }
```

#### 2. Code Formatting

```cpp
// Use 4 spaces for indentation (no tabs)
// Place opening brace on same line for functions/classes
class MyClass {
public:
    void myFunction() {
        if (condition) {
            doSomething();
        }
    }
};

// Maximum line length: 100 characters
// Use line breaks for long function signatures
void myLongFunctionName(
    const std::vector<Tensor<T>>& inputs,
    const std::vector<Tensor<T>>& targets,
    int epochs) {
    // implementation
}
```

#### 3. Include Order

```cpp
// 1. Corresponding header (for .cpp files)
#include "MyClass.hpp"

// 2. C++ standard library headers
#include <iostream>
#include <vector>
#include <memory>

// 3. Third-party library headers
// (none currently in SmartDNN)

// 4. Project headers
#include "smart_dnn/Tensor/Tensor.hpp"
#include "smart_dnn/Layer.hpp"
```

#### 4. Header Guards

```cpp
#ifndef MY_CLASS_HPP
#define MY_CLASS_HPP

// ... header content ...

#endif // MY_CLASS_HPP
```

#### 5. Comments

```cpp
// Use // for single-line comments
// Prefer explanatory comments over obvious ones

// Good:
// Compute gradient using chain rule
auto gradient = computeChainRule(input, output);

// Bad:
// Multiply by two
result = value * 2;

/**
 * Use /** */ for documentation comments
 * Document public APIs, complex algorithms, and non-obvious code
 * 
 * @param input Input tensor
 * @return Output tensor after transformation
 */
Tensor<T> forward(const Tensor<T>& input);
```

#### 6. Modern C++ Features

Use C++17 features:

```cpp
// Structured bindings
auto [inputs, targets] = loadData();

// if-init statements
if (auto result = compute(); result.isValid()) {
    // use result
}

// std::optional for nullable values
std::optional<Tensor<T>> weights;

// Template argument deduction
Tensor tensor(Shape{3, 3});  // Instead of Tensor<float>

// Fold expressions (for variadic templates)
template<typename... Args>
void addLayers(Args&&... args) {
    (addLayer(std::forward<Args>(args)), ...);
}
```

#### 7. Error Handling

```cpp
// Use exceptions for error handling
if (weights.size() != gradients.size()) {
    throw std::invalid_argument("Weights and gradients size mismatch!");
}

// Use std::runtime_error for runtime errors
if (computation_failed) {
    throw std::runtime_error("Failed to compute result");
}

// Document what exceptions can be thrown
/**
 * Computes forward pass
 * @throws std::invalid_argument if input shape is invalid
 * @throws std::runtime_error if computation fails
 */
Tensor<T> forward(const Tensor<T>& input);
```

#### 8. Templates

```cpp
// Use typename for template parameters
template <typename T>  // Good

template <class T>  // Avoid

// Provide default template arguments
template <typename T = float, typename DeviceType = CPUDevice>
class Tensor { };

// Use SFINAE or concepts (C++20) for template constraints
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, Tensor<T>>
createTensor() { }
```

#### 9. Memory Management

```cpp
// Prefer smart pointers over raw pointers
std::unique_ptr<Layer<T>> layer;  // Good
Layer<T>* layer;  // Avoid

// Use std::make_unique and std::make_shared
auto layer = std::make_unique<FullyConnectedLayer>(10, 20);

// Avoid manual new/delete
// Use RAII for resource management
```

---

## Making Changes

### Branch Naming

Use descriptive branch names:

```bash
# Features
git checkout -b feature/add-lstm-layer
git checkout -b feature/implement-adam-optimizer

# Bug fixes
git checkout -b fix/gradient-computation-bug
git checkout -b fix/memory-leak-in-conv2d

# Documentation
git checkout -b docs/update-api-reference
git checkout -b docs/add-tutorial

# Refactoring
git checkout -b refactor/optimize-tensor-operations
```

### Commit Messages

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add LSTM layer implementation"
git commit -m "Fix gradient computation in Conv2D layer"
git commit -m "Update API documentation for Tensor class"
git commit -m "Refactor optimizer interface for better extensibility"

# Bad commit messages (avoid these)
git commit -m "fix bug"
git commit -m "update"
git commit -m "wip"
```

**Commit Message Format:**

```
Short summary (50 chars or less)

More detailed explanation if necessary. Wrap at 72 characters.
Explain the problem this commit solves and why this approach
was chosen.

- Bullet points are okay too
- Use them to list multiple changes

Fixes #123
```

### Code Review Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New code has tests (if applicable)
- [ ] Documentation is updated
- [ ] No compiler warnings
- [ ] Code is properly commented
- [ ] Performance implications considered

---

## Testing Guidelines

### Running Tests

```bash
# Build tests
cd tests
cmake .
make

# Run all tests
./run_tests

# Run specific test suite
./test_tensor
./test_layers
```

### Writing Tests

Create tests in the `tests/` directory:

```cpp
// tests/test_my_feature.cpp
#include <iostream>
#include <cassert>
#include "smart_dnn/MyFeature.hpp"

void testBasicFunctionality() {
    // Arrange
    MyFeature feature(params);
    
    // Act
    auto result = feature.compute(input);
    
    // Assert
    assert(result == expected);
    std::cout << "Test passed: basic functionality" << std::endl;
}

void testEdgeCases() {
    // Test edge cases
    // ...
}

int main() {
    testBasicFunctionality();
    testEdgeCases();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
```

### Test Coverage

Aim to test:
- Normal operation
- Edge cases (empty inputs, single elements, etc.)
- Error conditions
- Boundary values
- Performance (for critical operations)

Example:

```cpp
void testTensorOperations() {
    // Normal operation
    Tensor<float> a(Shape{2, 2}, 1.0f);
    Tensor<float> b(Shape{2, 2}, 2.0f);
    auto c = a + b;
    assert(c[0] == 3.0f);
    
    // Edge case: empty tensor
    Tensor<float> empty(Shape{0});
    // Test behavior
    
    // Error condition
    try {
        Tensor<float> invalid(Shape{2, 2});
        Tensor<float> wrong(Shape{3, 3});
        auto result = invalid + wrong;  // Should throw
        assert(false && "Should have thrown exception");
    } catch (const std::exception&) {
        // Expected
    }
}
```

---

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Make sure your branch is up to date
git fetch upstream
git rebase upstream/main

# Run tests
cd tests
make && ./run_tests

# Check code style
# (use clang-format if available)
clang-format -i smart_dnn/**/*.hpp
```

### 2. Push to Your Fork

```bash
git push origin feature/my-new-feature
```

### 3. Create Pull Request

On GitHub:
1. Go to your fork
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out the PR template

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Changes Made
- List of changes
- Another change

## Testing
- [ ] All existing tests pass
- [ ] New tests added (if applicable)
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No compiler warnings
- [ ] Performance impact considered
```

### 4. Code Review

- Respond to reviewer comments
- Make requested changes
- Push updates to the same branch
- Be open to feedback

### 5. Merge

Once approved:
- Maintainers will merge your PR
- Your changes will be included in the next release
- Celebrate! 🎉

---

## Documentation

### Updating Documentation

When adding features, update:

1. **API Reference** (`docs/API_REFERENCE.md`)
   - Document new classes, methods, parameters
   - Provide code examples

2. **Tutorials** (`docs/TUTORIALS.md`)
   - Add tutorials for significant new features
   - Include complete, working examples

3. **Architecture Guide** (`docs/ARCHITECTURE.md`)
   - Explain architectural decisions
   - Update diagrams if structure changes

4. **README.md**
   - Update feature list
   - Add examples if applicable

### Documentation Style

```cpp
/**
 * Brief description of the function
 * 
 * Longer description with more details about what the function does,
 * how it works, and any important notes.
 * 
 * @tparam T Data type for tensor elements
 * @param input Input tensor with shape (batch_size, features)
 * @param weights Weight tensor with shape (features, output_size)
 * @return Output tensor with shape (batch_size, output_size)
 * 
 * @throws std::invalid_argument if shapes are incompatible
 * 
 * Example:
 * ```cpp
 * Tensor<float> input(Shape{32, 128});
 * Tensor<float> weights(Shape{128, 64});
 * auto output = matmul(input, weights);
 * // output.shape() == {32, 64}
 * ```
 */
template <typename T>
Tensor<T> matmul(const Tensor<T>& input, const Tensor<T>& weights);
```

---

## Reporting Bugs

### Before Reporting

1. Check existing issues
2. Verify you're using the latest version
3. Ensure it's reproducible

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## To Reproduce
Steps to reproduce:
1. Create model with...
2. Train on...
3. Error occurs...

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., Ubuntu 20.04]
- Compiler: [e.g., GCC 9.3]
- SmartDNN Version: [e.g., 1.0.0]

## Code Sample
```cpp
// Minimal code to reproduce the issue
```

## Additional Context
Any other relevant information
```

---

## Suggesting Features

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature

## Motivation
Why is this feature needed?
What problem does it solve?

## Proposed Solution
How would you implement this?

## Alternatives Considered
What other approaches did you consider?

## Additional Context
Examples, references, or other information

## Are you willing to implement this?
- [ ] Yes
- [ ] No
- [ ] With guidance
```

---

## Communication

### Getting Help

- **GitHub Issues:** For bug reports and feature requests
- **Email:** AndrewGeorgiou98@outlook.com for private inquiries
- **Discussions:** GitHub Discussions for general questions

### Response Times

- Issues: Typically reviewed within 1-3 days
- PRs: Initial review within 3-7 days
- Complex changes may take longer

---

## Recognition

Contributors will be:
- Listed in the contributors file
- Credited in release notes
- Mentioned in documentation (for significant contributions)

---

## License

By contributing to SmartDNN, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

If you have questions about contributing, feel free to:
1. Open an issue with the `question` label
2. Email the maintainer
3. Start a discussion on GitHub

Thank you for contributing to SmartDNN! 🚀

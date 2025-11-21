# Repository Graph Visualization

This directory contains a responsive, interactive visualization of the SmartDNN repository structure, components, and test results.

## Overview

The visualization provides:

- **Component Distribution**: Visual representation of all components (Activations, Layers, Optimizers, Loss Functions, Regularisation, Tensor operations, etc.)
- **Dependency Graph**: Interactive graph showing relationships between major components
- **Test Results Summary**: Comprehensive overview of all 105 tests across 11 test suites
- **Code Statistics**: Lines of code breakdown across headers, tests, and examples

## Generating the Visualization

To generate or update the visualization, run:

```bash
python3 generate_graph.py
```

This will:
1. Analyze the repository structure
2. Count lines of code in all components
3. Generate a responsive HTML file: `repository_graph.html`

## Viewing the Visualization

Simply open `repository_graph.html` in any modern web browser:

```bash
# On Linux/macOS
open repository_graph.html

# On Windows
start repository_graph.html

# Or use a simple HTTP server
python3 -m http.server 8080
# Then navigate to: http://localhost:8080/repository_graph.html
```

## Features

### Responsive Design
- Fully responsive layout that adapts to different screen sizes
- Mobile-friendly interface
- No external dependencies (all CSS/HTML inline)

### Interactive Elements
- Hover effects on all components
- Animated progress bars
- Color-coded component categories
- Interactive dependency graph

### Statistics Displayed

1. **Lines of Code**: Total non-comment lines across the codebase
2. **Components**: Number of major component categories
3. **Tests**: Total test count and pass rate
4. **Files**: Total number of header and implementation files

## Component Categories

The visualization organizes code into the following categories:

- **Activations** (Purple): ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Mish, Swish
- **Layers** (Pink): FullyConnectedLayer, Conv2DLayer, ActivationLayer, FlattenLayer
- **Optimizers** (Blue): AdamOptimizer, SGDOptimizer, RMSPropOptimizer
- **Loss Functions** (Green): MSELoss, CategoricalCrossEntropyLoss
- **Regularisation** (Red): DropoutLayer, BatchNormalizationLayer, MaxPooling2DLayer
- **Tensor** (Yellow): Core tensor operations and data structures
- **Datasets** (Cyan): Data loading utilities
- **Shape** (Orange): Shape manipulation utilities
- **Debugging** (Dark Red): Logging and debugging tools

## Requirements

- Python 3.6+
- No additional Python packages required (uses only standard library)
- Modern web browser for viewing the HTML output

## Customization

You can customize the visualization by editing `generate_graph.py`:

- Update colors in the `colors` dictionary
- Modify component categories
- Add additional statistics
- Customize the HTML template

## Test Coverage

The visualization shows test results from all test suites:

- AdvancedTensorOperationsTest (35 tests)
- DropoutLayerTest (3 tests)
- BatchNormalizationLayerTest (3 tests)
- TensorInitialisationTest (3 tests)
- TensorOperatorTest (8 tests)
- TensorCopyMoveTest (3 tests)
- TensorScalarOperatorTest (8 tests)
- ActivationTests (15 tests)
- FullyConnectedLayerTest (10 tests)
- Conv2DLayerTest (7 tests)
- OptimizerTests (10 tests)

**Total: 105/105 tests passing ✅**

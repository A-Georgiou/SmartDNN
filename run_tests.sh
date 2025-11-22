#!/bin/bash

# SmartDNN Test Runner Script
# This script builds and runs all tests for the SmartDNN library

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Print section header
print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo ""
}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TESTS_DIR="${SCRIPT_DIR}/tests"
BUILD_DIR="${TESTS_DIR}/build"

# Check if tests directory exists
if [ ! -d "$TESTS_DIR" ]; then
    print_message "$RED" "Error: Tests directory not found at ${TESTS_DIR}"
    exit 1
fi

print_header "SmartDNN Test Suite Runner"

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    print_message "$YELLOW" "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

# Navigate to build directory
cd "$BUILD_DIR"

# Configure with CMake
print_header "Configuring Tests with CMake"
if cmake .. ; then
    print_message "$GREEN" "✓ CMake configuration successful"
else
    print_message "$RED" "✗ CMake configuration failed"
    exit 1
fi

# Build the tests
print_header "Building Tests"
# Determine number of processors for parallel build (cross-platform)
if command -v nproc > /dev/null 2>&1; then
    NUM_PROCS=$(nproc)
elif command -v sysctl > /dev/null 2>&1; then
    NUM_PROCS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
    NUM_PROCS=4
fi

if make -j${NUM_PROCS} ; then
    print_message "$GREEN" "✓ Build successful"
else
    print_message "$RED" "✗ Build failed"
    exit 1
fi

# Run the tests
print_header "Running Tests"
if [ -f "./RunTests" ]; then
    if ./RunTests ; then
        print_message "$GREEN" "✓ All tests passed!"
        exit 0
    else
        print_message "$RED" "✗ Some tests failed"
        exit 1
    fi
else
    print_message "$RED" "✗ Test executable not found"
    exit 1
fi

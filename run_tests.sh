#!/bin/bash

# SmartDNN Test Runner Script
# This script builds and runs all tests for the SmartDNN project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="${SCRIPT_DIR}/tests"

# Check if tests directory exists
if [ ! -d "${TESTS_DIR}" ]; then
    echo -e "${RED}Error: Tests directory not found at ${TESTS_DIR}${NC}"
    exit 1
fi

# Parse command line arguments
CLEAN=false
VERBOSE=false
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            HELP=true
            shift
            ;;
    esac
done

# Display help
if [ "$HELP" = true ]; then
    echo "SmartDNN Test Runner"
    echo ""
    echo "Usage: ./run_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --clean      Clean build files before building"
    echo "  -v, --verbose    Run tests with verbose output"
    echo "  -h, --help       Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_tests.sh              # Run tests normally"
    echo "  ./run_tests.sh --clean      # Clean and rebuild before testing"
    echo "  ./run_tests.sh --verbose    # Run tests with detailed output"
    exit 0
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   SmartDNN Test Suite Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    cd "${TESTS_DIR}"
    
    # Files and directories to clean
    CLEAN_TARGETS=(
        "CMakeFiles/"
        "CMakeCache.txt"
        "cmake_install.cmake"
        "Makefile"
        "_deps/"
        "lib/"
        "Testing/"
        "RunTests"
        "CTestTestfile.cmake"
    )
    
    # Remove each target if it exists
    for target in "${CLEAN_TARGETS[@]}"; do
        if [ -e "$target" ]; then
            rm -rf "$target"
        fi
    done
    
    echo -e "${GREEN}✓ Clean complete${NC}"
    echo ""
fi

# Navigate to tests directory
cd "${TESTS_DIR}"

# Check for cmake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install cmake to build the tests${NC}"
    exit 1
fi

# Configure the build
echo -e "${YELLOW}Configuring test build with CMake...${NC}"
if ! cmake . > /dev/null 2>&1; then
    echo -e "${RED}✗ CMake configuration failed${NC}"
    echo -e "${RED}Running CMake with output for debugging:${NC}"
    cmake .
    exit 1
fi
echo -e "${GREEN}✓ Configuration complete${NC}"
echo ""

# Build the tests
echo -e "${YELLOW}Building test suite...${NC}"
if ! make > /dev/null 2>&1; then
    echo -e "${RED}✗ Build failed${NC}"
    echo -e "${RED}Running make with output for debugging:${NC}"
    make
    exit 1
fi
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# Run the tests
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Running Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$VERBOSE" = true ]; then
    # Run with verbose output
    if ctest --verbose; then
        TEST_RESULT=0
    else
        TEST_RESULT=$?
    fi
else
    # Run with standard output
    if ctest --output-on-failure; then
        TEST_RESULT=0
    else
        TEST_RESULT=$?
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"

# Display results
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}   ✓ All tests passed!${NC}"
else
    echo -e "${RED}   ✗ Some tests failed${NC}"
fi

echo -e "${BLUE}========================================${NC}"
echo ""

exit $TEST_RESULT

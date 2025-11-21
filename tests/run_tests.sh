#!/bin/bash

# SmartDNN Test Runner Script
# This script builds and runs the SmartDNN test suite

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Debug"
CLEAN_BUILD=false
RUN_SPECIFIC_TEST=""
VERBOSE=false
JOBS=$(nproc 2>/dev/null || echo 4)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -t|--test)
            RUN_SPECIFIC_TEST="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --clean           Clean build (remove CMake cache and build files)"
            echo "  -r, --release         Build in Release mode (default: Debug)"
            echo "  -t, --test FILTER     Run specific test(s) matching FILTER"
            echo "  -v, --verbose         Verbose test output"
            echo "  -j, --jobs N          Number of parallel build jobs (default: auto-detected, fallback to 4)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Build and run all tests"
            echo "  $0 -c                 # Clean build and run all tests"
            echo "  $0 -t Tensor*         # Run only Tensor tests"
            echo "  $0 -r -j8             # Release build with 8 parallel jobs"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Navigate to tests directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}SmartDNN Test Runner${NC}"
echo "===================="
echo ""

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    rm -rf CMakeCache.txt CMakeFiles cmake_install.cmake Makefile
    rm -rf _deps bin lib RunTests *.cmake
    echo ""
fi

# Configure CMake
echo -e "${YELLOW}Configuring CMake (${BUILD_TYPE})...${NC}"
cmake . -DCMAKE_BUILD_TYPE=$BUILD_TYPE
echo ""

# Build tests
echo -e "${YELLOW}Building tests with $JOBS parallel jobs...${NC}"
make -j$JOBS
echo ""

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
echo ""

TEST_CMD="./RunTests"

if [ -n "$RUN_SPECIFIC_TEST" ]; then
    TEST_CMD="$TEST_CMD --gtest_filter=$RUN_SPECIFIC_TEST"
fi

if [ "$VERBOSE" = true ]; then
    TEST_CMD="$TEST_CMD --gtest_verbose"
fi

# Run the tests and capture exit code
set +e
$TEST_CMD
TEST_EXIT_CODE=$?
set -e

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed (exit code: $TEST_EXIT_CODE)${NC}"
fi

exit $TEST_EXIT_CODE

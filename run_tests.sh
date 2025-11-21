#!/bin/bash

# SmartDNN Test Runner Script
# This script builds and runs all tests for the SmartDNN library

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "     SmartDNN Test Runner"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -d "tests" ]; then
    echo -e "${RED}Error: tests directory not found. Please run this script from the repository root.${NC}"
    exit 1
fi

# Navigate to tests directory
cd tests

# Clean previous builds if requested
if [ "$1" == "clean" ] || [ "$1" == "--clean" ]; then
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    make clean 2>/dev/null || true
    rm -rf CMakeCache.txt CMakeFiles/ Makefile cmake_install.cmake _deps/ bin/ lib/ RunTests 2>/dev/null || true
    echo ""
fi

# Configure with CMake
echo -e "${YELLOW}Configuring tests with CMake...${NC}"
cmake . > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    cmake .
    exit 1
fi
echo -e "${GREEN}✓ CMake configuration successful${NC}"
echo ""

# Build tests
echo -e "${YELLOW}Building tests...${NC}"
make -j$(nproc) 2>&1 | grep -E "(error|warning|Building|Linking)" || true
if [ $? -ne 0 ] && [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    make
    exit 1
fi
echo -e "${GREEN}✓ Build successful${NC}"
echo ""

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
echo ""
./RunTests

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================"
    echo -e "  All tests passed successfully! ✓"
    echo -e "======================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}======================================"
    echo -e "  Some tests failed! ✗"
    echo -e "======================================${NC}"
    exit 1
fi

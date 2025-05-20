#!/bin/bash
set -e

# Set ROCm path
export ROCM_PATH=/opt/rocm
echo "Using ROCm path: $ROCM_PATH"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake using ROCm path
echo "Configuring with CMake..."
ROCM_PATH=$ROCM_PATH cmake ..

# Build the test
echo "Building the test..."
make -j $(nproc)

# Run the test
echo "Running the test..."
./cumem_test 
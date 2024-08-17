cmake_minimum_required(VERSION 3.10)
project(SmartDNNTesting)

# Enable testing
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Include directories where your header files are located
include_directories(${PROJECT_SOURCE_DIR}/../smart_dnn/)

# Tensor tests
add_executable(test_tensor tensor/test_tensor.cpp
                           ../smart_dnn/Tensor/Tensor.cpp)

# Link the test executable against GTest libraries and your main library
target_link_libraries(test_tensor GTest::gtest GTest::gtest_main)

# Register the test
add_test(NAME TensorTests COMMAND test_tensor)

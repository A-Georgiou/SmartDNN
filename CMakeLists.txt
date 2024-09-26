cmake_minimum_required(VERSION 3.10)
project(SmartDNN)
set(CMAKE_CXX_STANDARD 17)

option(USE_CPU_TENSORS "Run SmartDNN without any additional libraries" ON)
option(USE_ARRAYFIRE_TENSORS "Run SmartDNN with ArrayFire tensors" ON)

# Find ArrayFire package
if (USE_ARRAYFIRE_TENSORS)
    find_package(ArrayFire REQUIRED)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "/W4")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
endif()

include_directories(${PROJECT_SOURCE_DIR})

add_executable(SmartDNN
    src/main.cpp
    smart_dnn/tensor/TensorBase.cpp
    smart_dnn/tensor/TensorAdapterBase.cpp
    smart_dnn/tensor/TensorBackendUtil.cpp
    smart_dnn/tensor/Backend/Default/CPUTensor.cpp
    smart_dnn/tensor/Backend/Default/CPUTensorBackend.cpp
)

# Link ArrayFire library to SmartDNN executable
target_link_libraries(SmartDNN ArrayFire::af)

# Enable logging for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(SmartDNN PRIVATE ENABLE_LOGGING)
endif()

# Parallel build setup
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CTEST_BUILD_FLAGS -j${N})
    set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${N})
endif()
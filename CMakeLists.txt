cmake_minimum_required(VERSION 3.10)
project(SmartDNN)
set(CMAKE_CXX_STANDARD 17)

option(USE_CPU_TENSORS "Run SmartDNN without any additional libraries" OFF)
option(USE_ARRAYFIRE_TENSORS "Run SmartDNN with ArrayFire tensors" ON)

if (USE_ARRAYFIRE_TENSORS)
    set(USE_CPU_TENSORS OFF)
    message(STATUS "ARRAYFIRE TENSORS ENABLED")
    find_package(ArrayFire REQUIRED)
    add_definitions(-DUSE_ARRAYFIRE_TENSORS)
elseif (USE_CPU_TENSORS)
    set(USE_ARRAYFIRE_TENSORS OFF)
    message(STATUS "CPU TENSORS ENABLED")
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

# Common source files
set(COMMON_SOURCES
    src/main.cpp
    smart_dnn/tensor/TensorBase.cpp
    smart_dnn/tensor/TensorAdapterBase.cpp
    smart_dnn/tensor/TensorBackendUtil.cpp
)

# Conditional source files
if (USE_ARRAYFIRE_TENSORS)
    set(TENSOR_SOURCES
        smart_dnn/tensor/Backend/ArrayFire/GPUTensor.cpp
        smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.cpp
        smart_dnn/tensor/Backend/ArrayFire/Utils.cpp
    )
elseif (USE_CPU_TENSORS)
    set(TENSOR_SOURCES
        smart_dnn/tensor/Backend/Default/CPUTensor.cpp
        smart_dnn/tensor/Backend/Default/CPUTensorBackend.cpp
    )
endif()

# Create the executable with all source files
add_executable(SmartDNN ${COMMON_SOURCES} ${TENSOR_SOURCES})

if (USE_ARRAYFIRE_TENSORS)
    target_link_libraries(SmartDNN ArrayFire::af)
endif()

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
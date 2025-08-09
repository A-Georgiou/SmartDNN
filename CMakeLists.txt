cmake_minimum_required(VERSION 3.10)
project(SmartDNN)
set(CMAKE_CXX_STANDARD 17)

option(USE_CPU_TENSORS "Run SmartDNN without any additional libraries" OFF)
option(USE_ARRAYFIRE_TENSORS "Run SmartDNN with ArrayFire tensors" OFF)
option(USE_EIGEN_TENSORS "Run SmartDNN with Eigen tensors" ON)

if (USE_ARRAYFIRE_TENSORS)
    set(USE_CPU_TENSORS OFF)
    set(USE_EIGEN_TENSORS OFF)
    message(STATUS "ARRAYFIRE TENSORS ENABLED")
    find_package(ArrayFire REQUIRED)
    add_definitions(-DUSE_ARRAYFIRE_TENSORS)
elseif (USE_EIGEN_TENSORS)
    set(USE_CPU_TENSORS OFF)
    set(USE_ARRAYFIRE_TENSORS OFF)
    message(STATUS "EIGEN TENSORS ENABLED")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(EIGEN3 REQUIRED eigen3)
    add_definitions(-DUSE_EIGEN_TENSORS)
elseif (USE_CPU_TENSORS)
    set(USE_ARRAYFIRE_TENSORS OFF)
    set(USE_EIGEN_TENSORS OFF)
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
elseif (USE_EIGEN_TENSORS)
    set(TENSOR_SOURCES
        smart_dnn/tensor/Backend/Default/CPUTensor.cpp
        smart_dnn/tensor/Backend/Default/CPUTensorBackend.cpp
        smart_dnn/tensor/Backend/Eigen/EigenTensorBackend.cpp
    )
elseif (USE_CPU_TENSORS)
    set(TENSOR_SOURCES
        smart_dnn/tensor/Backend/Default/CPUTensor.cpp
        smart_dnn/tensor/Backend/Default/CPUTensorBackend.cpp
    )
endif()

# Create a library with all source files
add_library(SmartDNNLib ${COMMON_SOURCES} ${TENSOR_SOURCES})

if (USE_ARRAYFIRE_TENSORS)
    target_link_libraries(SmartDNNLib ArrayFire::af)
elseif (USE_EIGEN_TENSORS)
    target_include_directories(SmartDNNLib PRIVATE ${EIGEN3_INCLUDE_DIRS})
    target_compile_definitions(SmartDNNLib PRIVATE ${EIGEN3_CFLAGS_OTHER})
endif()

# Enable logging for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(SmartDNNLib PRIVATE ENABLE_LOGGING)
endif()

# Add example executables
add_executable(SimpleLinearRegression examples/SimpleLinearRegressionModel.cpp)
target_link_libraries(SimpleLinearRegression SmartDNNLib)

add_executable(MNist examples/MNistModel.cpp)
target_link_libraries(MNist SmartDNNLib)

if (USE_ARRAYFIRE_TENSORS)
    target_link_libraries(SimpleLinearRegression ArrayFire::af)
    target_link_libraries(MNist ArrayFire::af)
elseif (USE_EIGEN_TENSORS)
    target_include_directories(SimpleLinearRegression PRIVATE ${EIGEN3_INCLUDE_DIRS})
    target_compile_definitions(SimpleLinearRegression PRIVATE ${EIGEN3_CFLAGS_OTHER})
    target_include_directories(MNist PRIVATE ${EIGEN3_INCLUDE_DIRS})
    target_compile_definitions(MNist PRIVATE ${EIGEN3_CFLAGS_OTHER})
endif()

# Parallel build setup
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CTEST_BUILD_FLAGS -j${N})
    set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${N})
endif()
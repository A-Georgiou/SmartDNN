cmake_minimum_required(VERSION 3.10)
project(SmartDNNTesting)

set(CMAKE_CXX_STANDARD 17)

option(USE_CPU_TENSORS "Run SmartDNN without any additional libraries" ON)
option(USE_ARRAYFIRE_TENSORS "Run SmartDNN with ArrayFire tensors" OFF)

if (USE_ARRAYFIRE_TENSORS)
    set(USE_CPU_TENSORS OFF)
    message(STATUS "ARRAYFIRE TENSORS ENABLED")
    find_package(ArrayFire REQUIRED)
    add_definitions(-DUSE_ARRAYFIRE_TENSORS)
elseif (USE_CPU_TENSORS)
    set(USE_ARRAYFIRE_TENSORS OFF)
    message(STATUS "CPU TENSORS ENABLED")
endif()

enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(${PROJECT_SOURCE_DIR}/../)

file(GLOB_RECURSE TEST_SOURCES main.cpp **/*.cpp)

set(COMMON_SOURCES
    ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/TensorBase.cpp
    ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/TensorAdapterBase.cpp
    ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/TensorBackendUtil.cpp
)

if (USE_ARRAYFIRE_TENSORS)
    set(TENSOR_SOURCES
        ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/Backend/ArrayFire/GPUTensor.cpp
        ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/Backend/ArrayFire/GPUTensorBackend.cpp
        ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/Backend/ArrayFire/Utils.cpp
    )
elseif (USE_CPU_TENSORS)
    set(TENSOR_SOURCES
    ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/Backend/Default/CPUTensor.cpp
        ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/Backend/Default/CPUTensorBackend.cpp
    )
endif()

list(FILTER TEST_SOURCES EXCLUDE REGEX ".*CMakeFiles/.*.cpp$")

message("Test Sources: ${TEST_SOURCES}")

add_executable(RunTests ${TEST_SOURCES} ${COMMON_SOURCES} ${TENSOR_SOURCES})

if (USE_ARRAYFIRE_TENSORS)
    target_link_libraries(RunTests ArrayFire::af)
endif()

target_link_libraries(RunTests GTest::gtest GTest::gtest_main)

# Register the test
add_test(NAME TensorTests COMMAND RunTests)

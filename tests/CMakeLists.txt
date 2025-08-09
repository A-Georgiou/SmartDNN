cmake_minimum_required(VERSION 3.10)
project(SmartDNNTesting)

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
elseif (USE_EIGEN_TENSORS)
    set(TENSOR_SOURCES
        ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/Backend/Default/CPUTensor.cpp
        ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/Backend/Eigen/EigenTensorBackend.cpp
    )
elseif (USE_CPU_TENSORS)
    set(TENSOR_SOURCES
    ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/Backend/Default/CPUTensor.cpp
        ${PROJECT_SOURCE_DIR}/../smart_dnn/tensor/Backend/Default/CPUTensorBackend.cpp
    )
endif()

list(FILTER TEST_SOURCES EXCLUDE REGEX ".*CMakeFiles/.*.cpp$")

if (USE_ARRAYFIRE_TENSORS)
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/tensor/test_templated_functions.cpp$")
endif()


add_executable(RunTests ${TEST_SOURCES} ${COMMON_SOURCES} ${TENSOR_SOURCES})

if (USE_ARRAYFIRE_TENSORS)
    target_link_libraries(RunTests ArrayFire::af)
elseif (USE_EIGEN_TENSORS)
    target_include_directories(RunTests PRIVATE ${EIGEN3_INCLUDE_DIRS})
    target_compile_definitions(RunTests PRIVATE ${EIGEN3_CFLAGS_OTHER})
endif()

target_link_libraries(RunTests GTest::gtest GTest::gtest_main)

# Register the test
add_test(NAME TensorTests COMMAND RunTests)

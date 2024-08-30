cmake_minimum_required(VERSION 3.10)
project(SmartDNN)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(smart_dnn)

add_executable(SmartDNN
    smart_dnn/Tensor/Tensor.cpp
    smart_dnn/SmartDNN/SmartDNN.cpp
    src/main.cpp
)

# Enable logging for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(SmartDNN PRIVATE ENABLE_LOGGING)
endif()

include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CTEST_BUILD_FLAGS -j${N})
    set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${N})
endif()
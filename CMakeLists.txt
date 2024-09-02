cmake_minimum_required(VERSION 3.10)
project(SmartDNN)
set(CMAKE_CXX_STANDARD 17)

include_directories(smart_dnn)

add_executable(SmartDNN
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
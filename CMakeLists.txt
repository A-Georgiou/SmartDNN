cmake_minimum_required(VERSION 3.10)
project(SmartDNN)

set(CMAKE_CXX_STANDARD 17)

include_directories(smart_dnn)

add_executable(SmartDNN
    smart_dnn/Tensor/Tensor.cpp
    src/main.cpp
)

# Enable logging for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(SmartDNN PRIVATE ENABLE_LOGGING)
endif()
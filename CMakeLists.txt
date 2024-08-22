cmake_minimum_required(VERSION 3.10)
project(SmartDNN)
set(CMAKE_CXX_STANDARD 17)

include_directories(smart_dnn)

add_executable(SmartDNN
    smart_dnn/Tensor/Tensor.cpp
    smart_dnn/SmartDNN/SmartDNN.cpp
    src/main.cpp
)

option(USE_OPENBLAS "Download and use OpenBLAS" OFF)

if(USE_OPENBLAS)
    include(FetchContent)

    FetchContent_Declare(
        openblas
        URL https://github.com/xianyi/OpenBLAS/archive/v0.3.28.tar.gz 
    )

    FetchContent_MakeAvailable(openblas)

    target_include_directories(${PROJECT_NAME} PRIVATE ${openblas_SOURCE_DIR} ${openblas_BINARY_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${openblas_BINARY_DIR}/libopenblas.a)

    message(STATUS "OpenBLAS has been downloaded and linked.")
else()
    find_package(OpenBLAS REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS::OpenBLAS)
    message(STATUS "Using system-installed OpenBLAS.")
endif()



# Enable logging for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(SmartDNN PRIVATE ENABLE_LOGGING)
endif()
cmake_minimum_required(VERSION 3.10)
project(SmartDNN)
set(CMAKE_CXX_STANDARD 17)

include_directories(smart_dnn)

add_executable(SmartDNN
    smart_dnn/Tensor/Tensor.cpp
    smart_dnn/SmartDNN/SmartDNN.cpp
    src/main.cpp
)

# Check if OpenBLAS exists in the external directory
set(OPENBLAS_DIR ${CMAKE_SOURCE_DIR}/external/OpenBLAS)
if(EXISTS ${OPENBLAS_DIR})
    message(STATUS "OpenBLAS found in ${OPENBLAS_DIR}")
    
    # Check if OpenBLAS is already built
    if(NOT EXISTS ${OPENBLAS_DIR}/libopenblas.a)
        message(STATUS "OpenBLAS library not found. Building OpenBLAS...")
        
        # Build OpenBLAS
        execute_process(
            COMMAND make DYNAMIC_ARCH=1 NO_SHARED=1 NO_LAPACK=1 USE_OPENMP=0
            WORKING_DIRECTORY ${OPENBLAS_DIR}
            RESULT_VARIABLE OPENBLAS_BUILD_RESULT
        )
        
        if(NOT OPENBLAS_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to build OpenBLAS. Please build it manually.")
        endif()
    endif()
    
    # Add OpenBLAS include directory
    target_include_directories(SmartDNN PRIVATE ${OPENBLAS_DIR})
    
    # Add OpenBLAS library
    target_link_libraries(SmartDNN PRIVATE ${OPENBLAS_DIR}/libopenblas.a)
    
    # Define a preprocessor macro to indicate OpenBLAS is available
    target_compile_definitions(SmartDNN PRIVATE USE_OPENBLAS)
else()
    message(STATUS "OpenBLAS not found in ${OPENBLAS_DIR}. Building without OpenBLAS support.")
endif()

# Enable logging for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(SmartDNN PRIVATE ENABLE_LOGGING)
endif()

# Enable parallel build
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CTEST_BUILD_FLAGS -j${N})
    set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${N})
endif()
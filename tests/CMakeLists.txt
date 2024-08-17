cmake_minimum_required(VERSION 3.10)
project(SmartDNNTesting)

set(CMAKE_CXX_STANDARD 17)

# Enable testing
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Include directories where your header files are located
include_directories(${PROJECT_SOURCE_DIR}/../smart_dnn/)

file(GLOB_RECURSE TEST_SOURCES ${PROJECT_SOURCE_DIR}/**/*.cpp)

file(GLOB_RECURSE MAIN_SOURCES ${PROJECT_SOURCE_DIR}/../smart_dnn/**/*.cpp)

# Add executable that will run the tests, combining both main sources and test sources
add_executable(RunTests ${TEST_SOURCES} ${MAIN_SOURCES})

target_link_libraries(RunTests GTest::gtest GTest::gtest_main)

# Register the test
add_test(NAME TensorTests COMMAND RunTests)

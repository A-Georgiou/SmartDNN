cmake_minimum_required(VERSION 3.10)
project(SmartDNNTesting)

set(CMAKE_CXX_STANDARD 17)

enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(${PROJECT_SOURCE_DIR}/../smart_dnn/)

file(GLOB_RECURSE TEST_SOURCES main.cpp **/*.cpp)

file(GLOB_RECURSE MAIN_SOURCES ${PROJECT_SOURCE_DIR}/../smart_dnn/**/*.cpp)

list(FILTER TEST_SOURCES EXCLUDE REGEX ".*CMakeFiles/.*.cpp$")

message("Test Sources: ${TEST_SOURCES}")
message("Main Sources: ${MAIN_SOURCES}")

add_executable(RunTests ${TEST_SOURCES} ${MAIN_SOURCES})

target_link_libraries(RunTests GTest::gtest GTest::gtest_main)

# Register the test
add_test(NAME TensorTests COMMAND RunTests)

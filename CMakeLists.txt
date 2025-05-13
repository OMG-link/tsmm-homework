cmake_minimum_required(VERSION 3.10)
project(Matmul)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -Wall -Wextra -fsanitize=undefined,address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

file(GLOB LIBMATMUL_SOURCES "src/matmul.cpp" "src/core/matmul.c" "src/core/pack.c")
add_library(libmatmul STATIC ${LIBMATMUL_SOURCES})
target_include_directories(libmatmul PUBLIC include)

add_subdirectory(test/tests)

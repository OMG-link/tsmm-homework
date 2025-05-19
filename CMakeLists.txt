cmake_minimum_required(VERSION 3.10)
project(Matmul)

include(CheckCXXCompilerFlag)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)
add_compile_options(-fmax-errors=3)

check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)

if(COMPILER_SUPPORTS_AVX512F)
    add_compile_options(-mavx512f)
else()
    message(STATUS "Compiler does NOT support -mavx512f, skipping AVX-512")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

set(CMAKE_C_FLAGS_DEBUG     "-O0 -g -Wall -Wextra -fsanitize=undefined,address")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -Wall -Wextra -fsanitize=undefined,address")
set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

file(GLOB LIBMATMUL_SOURCES
    "src/matmul.cpp"
    "src/utils.c"
    "src/core/matmul.c"
    "src/core/pack.c"
    "src/core/kernels/*.cpp"
)

add_library(libmatmul STATIC ${LIBMATMUL_SOURCES})
target_include_directories(libmatmul PUBLIC include)
target_link_libraries(libmatmul PUBLIC OpenMP::OpenMP_CXX)

add_subdirectory(test/tests)

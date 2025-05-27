file(GLOB TEST_SOURCES "func*.cpp" "perf*.cpp")

foreach(test_source ${TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME_WE)

  add_executable(${test_name} ${test_source})
  target_link_libraries(${test_name} PRIVATE libmatmul)

  add_test(
    NAME ${test_name}
    COMMAND ${test_name}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
endforeach()

# mkl
find_package(OpenMP REQUIRED)

add_executable(ext_mkl "ext_mkl.cpp")
target_link_libraries(ext_mkl
  PRIVATE
    libmatmul
    mkl_intel_lp64
    mkl_sequential
    mkl_core
    pthread
    m
    dl
  PUBLIC
    OpenMP::OpenMP_CXX
)

# tune
add_executable(tune_blocking_strategy "tune_blocking_strategy.cpp")
target_link_libraries(tune_blocking_strategy PRIVATE libmatmul)

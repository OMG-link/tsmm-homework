file(GLOB TEST_SOURCES "func*.cpp" "perf*.cpp" "tune_blocking_strategy.cpp")

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

set(ext_mkl_source "ext_mkl.cpp")
add_executable(ext_mkl ${ext_mkl_source})
target_link_libraries(ext_mkl
  PRIVATE
    mkl_intel_lp64
    mkl_sequential
    mkl_core
    pthread
    m
    dl
  PUBLIC
    OpenMP::OpenMP_CXX
)
add_test(
  NAME ext_mkl
  COMMAND ext_mkl
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
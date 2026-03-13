cmake_minimum_required(VERSION 3.20)
project(Field VERSION 0.1.0 LANGUAGES CXX)

option(FIELD_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(field INTERFACE)
target_include_directories(field INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)


if(FIELD_BUILD_TESTS)
  enable_testing()

  function(field_add_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE field)
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME ${target_name} COMMAND ${target_name})
  endfunction()

  field_add_test(vector_test tests/vector_test.cc)
  field_add_test(matrix_test tests/matrix_test.cc)
endif()

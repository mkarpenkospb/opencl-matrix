cmake_minimum_required(VERSION 3.1)
project(opencl_matrix)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(libs)


add_executable(${PROJECT_NAME} src/main.cpp src/cl/naive.cl src/cl/with_tiles.cl src/matrix_mult.cpp)
target_link_libraries(${PROJECT_NAME} libclew libutils)

cmake_minimum_required(VERSION 3.10)
project(matmul)

set(CMAKE_CXX_STANDARD 11)

add_executable(sequential sequential.cpp)

add_executable(tiling sequential.cpp)
target_compile_definitions(tiling PRIVATE TILING)

add_executable(slow sequential.cpp)
target_compile_definitions(slow PRIVATE SLOW)

add_executable(avx sequential.cpp)
target_compile_definitions(avx PRIVATE avx)


cmake_minimum_required(VERSION 3.10)
project(matmul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

add_executable(sequential sequential.cpp)

add_executable(tiling sequential.cpp)
target_compile_definitions(tiling PRIVATE TILING)

add_executable(slow sequential.cpp)
target_compile_definitions(slow PRIVATE SLOW)

add_executable(avx sequential.cpp)
target_compile_definitions(avx PRIVATE avx)

add_executable(naive matmul.cu)
set_property(TARGET naive PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(shared matmul.cu)
target_compile_definitions(shared PRIVATE SHARED)
set_property(TARGET shared PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(cublas matmul.cu)
target_compile_definitions(cublas PRIVATE CUBLAS)
set_property(TARGET cublas PROPERTY CUDA_ARCHITECTURES OFF)
target_link_libraries(cublas -lcublas)

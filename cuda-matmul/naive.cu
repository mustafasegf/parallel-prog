#include "matrix.cpp"
#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <stdint.h>
#include <stdio.h>

__global__ void matrixMulKernel(const int32_t *matrix1, const int32_t *matrix2,
                                int32_t *answer, int32_t rows1, int32_t cols1,
                                int32_t cols2) {

  int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t sum = 0;

  if (row < rows1 && col < cols2) {
    for (int32_t i = 0; i < cols1; i++) {
      sum += matrix1[row * cols1 + i] * matrix2[i * cols2 + col];
    }
    answer[row * cols2 + col] = sum;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <matrix file> <matrix file>"
              << std::endl;
    return 1;
  }

  try {
    Matrix<int32_t> matrix1(argv[1]);
    Matrix<int32_t> matrix2(argv[2]);

    Matrix<int32_t> answer(matrix1.rows, matrix2.cols);

    auto start = std::chrono::high_resolution_clock::now();
    int *device_1, *device_2, *device_answer;

    // auto start_alloc = std::chrono::high_resolution_clock::now();
    // allocate device memory

    cudaMalloc(&device_1, matrix1.rows * matrix1.cols * sizeof(int32_t));
    cudaMalloc(&device_2, matrix2.rows * matrix2.cols * sizeof(int32_t));
    cudaMalloc(&device_answer, matrix1.rows * matrix2.cols * sizeof(int32_t));

    // auto end_alloc = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::micro> duration_alloc =
    //     end_alloc - start_alloc;
    // std::cout << "alloc us: " << duration_alloc.count() << std::endl;

    // copy data to device

    // auto start_copy = std::chrono::high_resolution_clock::now();
    cudaMemcpy(device_1, matrix1.begin(),
               matrix1.rows * matrix1.cols * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    cudaMemcpy(device_2, matrix2.begin(),
               matrix2.rows * matrix2.cols * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    // auto end_copy = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::micro> duration_copy =
    //     end_copy - start_copy;
    // std::cout << "copy us: " << duration_copy.count() << std::endl;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((matrix2.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (matrix1.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // std::cout << "numBlocks: " << numBlocks.x << "x" << numBlocks.y
    //           << std::endl;
    // std::cout << "threadsPerBlock: " << threadsPerBlock.x << "x"
    //           << threadsPerBlock.y << std::endl;

    // start calculation
    // auto start_compute = std::chrono::high_resolution_clock::now();
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(device_1, device_2,
                                                    device_answer, matrix1.rows,
                                                    matrix1.cols, matrix2.cols);

    // auto end_compute = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::micro> duration_compute =
    //     end_compute - start_compute;
    // std::cout << "compute us: " << duration_compute.count() << std::endl;

    // copy data back to host
    // auto start_comm = std::chrono::high_resolution_clock::now();
    cudaMemcpy(answer.begin(), device_answer,
               matrix1.rows * matrix2.cols * sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    // auto end_comm = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::micro> duration_comm =
    //     end_comm - start_comm;
    // std::cout << "comm us: " << duration_comm.count() << std::endl;

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> duration = end - start;

    std::cout << "size: " << matrix1.rows << "x" << matrix2.cols
              << " compute us: " << duration.count() << " comm us: 0"
              << std::endl;

    // std::cout << answer << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}

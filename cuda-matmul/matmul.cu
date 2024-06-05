#include "matrix.hpp"
#include <cassert>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdint.h>
#include <stdio.h>

using data_type = double;

__global__ void matrixMulKernel(const data_type *matrix1,
                                const data_type *matrix2, data_type *answer,
                                int32_t n) {

  int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  data_type sum = 0;

  if (row < n && col < n) {
    for (int32_t i = 0; i < n; i++) {
      sum += matrix1[row * n + i] * matrix2[i * n + col];
    }
    answer[row * n + col] = sum;
  }
}

constexpr int32_t BLOCK_SIZE = 32;

__global__ void gpu_square_matrix_mult(data_type *d_a, data_type *d_b,
                                       data_type *d_result, int n) {
  __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int tmp = 0;
  int idx;

  for (int sub = 0; sub < gridDim.x; ++sub) {
    idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
    if (idx >= n * n) {
      // n may not divisible by BLOCK_SIZE
      tile_a[threadIdx.y][threadIdx.x] = 0;
    } else {
      tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
    }

    idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
    if (idx >= n * n) {
      tile_b[threadIdx.y][threadIdx.x] = 0;
    } else {
      tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
    }
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < n && col < n) {
    d_result[row * n + col] = tmp;
  }
}

// constexpr int32_t TILE_WIDTH = 32;
// __global__ void matrixMulSharedKernel(const data_type *matrix1,
//                                       const data_type *matrix2,
//                                       data_type *answer, int32_t n) {
//
//   __shared__ data_type shared_matrix1[TILE_WIDTH][TILE_WIDTH];
//   __shared__ data_type shared_matrix2[TILE_WIDTH][TILE_WIDTH];
//
//   int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
//   int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
//
//   data_type tmp = 0;
//   int32_t idx;
//   data_type sum = 0;
//
//   shared_matrix1[threadIdx.y][threadIdx.x] = 0;
//   shared_matrix2[threadIdx.y][threadIdx.x] = 0;
//   if (row < n && col < n) {
//     for (int32_t i = 0; i < cols1; i += TILE_WIDTH) {
//       shared_matrix1[threadIdx.y][threadIdx.x] =
//           matrix1[row * cols1 + i + threadIdx.x];
//       shared_matrix2[threadIdx.y][threadIdx.x] =
//           matrix2[(i + threadIdx.y) * cols2 + col];
//       __syncthreads();
//
//       for (int32_t j = 0; j < TILE_WIDTH; j++) {
//         sum += shared_matrix1[threadIdx.y][j] * shared_matrix2[j][threadIdx.x];
//       }
//       __syncthreads();
//     }
//     answer[row * cols2 + col] = sum;
//   }
// }

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <matrix file> <matrix file>"
              << std::endl;
    return 1;
  }

  try {
    Matrix<data_type> matrix1(argv[1]);
    Matrix<data_type> matrix2(argv[2]);

    Matrix<data_type> answer(matrix1.rows, matrix2.cols);

    auto start = std::chrono::high_resolution_clock::now();
    data_type *device_1, *device_2, *device_answer;

    // auto start_alloc = std::chrono::high_resolution_clock::now();
    // allocate device memory

    cudaMalloc(&device_1, matrix1.rows * matrix1.cols * sizeof(data_type));
    cudaMalloc(&device_2, matrix2.rows * matrix2.cols * sizeof(data_type));
    cudaMalloc(&device_answer, matrix1.rows * matrix2.cols * sizeof(data_type));

    // auto end_alloc = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::micro> duration_alloc =
    //     end_alloc - start_alloc;
    // std::cout << "alloc us: " << duration_alloc.count() << std::endl;

    // copy data to device

    // auto start_copy = std::chrono::high_resolution_clock::now();
    cudaMemcpy(device_1, matrix1.begin(),
               matrix1.rows * matrix1.cols * sizeof(data_type),
               cudaMemcpyHostToDevice);

    cudaMemcpy(device_2, matrix2.begin(),
               matrix2.rows * matrix2.cols * sizeof(data_type),
               cudaMemcpyHostToDevice);

    // auto end_copy = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::micro> duration_copy =
    //     end_copy - start_copy;
    // std::cout << "copy us: " << duration_copy.count() << std::endl;

    auto grid = 32;
    if (argc > 3) {
      grid = std::stoi(argv[3]);
      if (grid == 0) {
        grid = 32;
      }
    }

    auto block = 32;
    if (argc > 4) {
      block = std::stoi(argv[4]);
      if (block == 0) {
        block = 32;
      }
    }

    dim3 gridSize((matrix2.cols + block - 1) / block,
                  (matrix1.rows + block - 1) / block);
    dim3 blockSize(block, block);

    // start calculation
    // auto start_compute = std::chrono::high_resolution_clock::now();

#ifdef SHARED
    auto name = "shared";
    // matrixMulSharedKernel<<<gridSize, blockSize>>>(device_1, device_2,
    gpu_square_matrix_mult<<<gridSize, blockSize>>>(
        device_1, device_2, device_answer, matrix1.rows);

#elif defined CUBLAS
    auto name = "cublas";

    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    data_type alpha = 1.0f;
    data_type beta = 0.0f;

    cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, matrix1.rows, matrix2.cols,
                matrix1.cols, &alpha, device_1, matrix1.cols, device_2,
                matrix2.rows, &beta, device_answer, matrix1.rows);
#else
    auto name = "naive";
    matrixMulKernel<<<gridSize, blockSize>>>(device_1, device_2, device_answer,
                                             matrix1.rows);

#endif

    // auto end_compute = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::micro> duration_compute =
    //     end_compute - start_compute;
    // std::cout << "compute us: " << duration_compute.count() << std::endl;

    // copy data back to host
    // auto start_comm = std::chrono::high_resolution_clock::now();
    cudaMemcpy(answer.begin(), device_answer,
               matrix1.rows * matrix2.cols * sizeof(data_type),
               cudaMemcpyDeviceToHost);

    // auto end_comm = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::micro> duration_comm =
    //     end_comm - start_comm;
    // std::cout << "comm us: " << duration_comm.count() << std::endl;

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> duration = end - start;

    auto fileName = std::string(argv[1]);
    size_t pos = fileName.rfind(".txt");
    if (pos == std::string::npos) {
      pos = fileName.length(); // Append at the end if ".txt" is not found
    }

    fileName.insert(pos, "_ans");
    Matrix<data_type> res(fileName);

    std::cout << answer << std::endl;

    for (size_t i = 0; i < answer.rows; i++) {
      for (size_t j = 0; j < answer.cols; j++) {
        assert(answer(i, j) == res(i, j));
      }
    }

    auto size =
        std::to_string(matrix1.rows) + "x" + std::to_string(matrix2.cols);

    std::cout << std::fixed << std::setprecision(0) << std::left << std::setw(6)
              << name << " " << std::setw(11) << size << " "
              << "grid: " << std::setw(3) << grid << " block: " << std::setw(5)
              << block << " " << duration.count() << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}

#include "matrix.cpp"
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <stdint.h>

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

#if defined(FAST)
    // fast
    for (size_t i = 0; i < matrix1.rows; i++) {
      for (size_t k = 0; k < matrix1.cols; k++) {
        answer(i, k) = 0;
        for (size_t j = 0; j < matrix2.cols; j++) {
          answer(i, j) += matrix1(i, k) * matrix2(k, j);
        }
      }
    }

#elif defined(TILING)
    // loop tiling
    size_t blockSize = 64;
    for (size_t i0 = 0; i0 < matrix1.rows; i0 += blockSize) {
      for (size_t j0 = 0; j0 < matrix2.cols; j0 += blockSize) {
        for (size_t k0 = 0; k0 < matrix1.cols; k0 += blockSize) {
          for (size_t i = i0; i < std::min(i0 + blockSize, matrix1.rows); i++) {
            for (size_t k = k0; k < std::min(k0 + blockSize, matrix1.cols);
                 k++) {
              for (size_t j = j0; j < std::min(j0 + blockSize, matrix2.cols);
                   j++) {
                answer(i, j) += matrix1(i, k) * matrix2(k, j);
              }
            }
          }
        }
      }
    }

#elif defined(AVX)
    // avx512
    const size_t blockSize = 512 / sizeof(matrix1[0][0]);
    for (size_t i = 0; i < matrix1.rows; i++) {
      for (size_t j = 0; j < matrix2.cols; j += blockSize) {
        __m512i sum_vector = _mm512_setzero_si512();

        for (size_t k = 0; k < matrix1.cols; k++) {
          __m512i mat1_vec = _mm512_set1_epi32(matrix1(i, k));

          __m512i mat2_vec = _mm512_load_si512((__m512i *)&matrix2(k, j));
          __m512i product_vec = _mm512_mullo_epi32(mat1_vec, mat2_vec);
          sum_vector = _mm512_add_epi32(sum_vector, product_vec);
        }

        // Store the results back to the answer matrix
        _mm512_store_si512((__m512i *)&answer(i, j), sum_vector);
      }
    }

#else
    // slow
    for (size_t i = 0; i < matrix1.rows; i++) {
      for (size_t j = 0; j < matrix2.cols; j++) {
        answer(i, j) = 0;
        for (size_t k = 0; k < matrix1.cols; k++) {
          answer(i, j) += matrix1(i, k) * matrix2(k, j);
        }
      }
    }

#endif

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

  return 0;
}

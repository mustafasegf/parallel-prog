#include "matrix.hpp"
#include <chrono>
#include <immintrin.h>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <matrix file> <matrix file>"
              << std::endl;
    return 1;
  }

  matrix_struct *m_1 = get_matrix_struct(argv[1]);
  matrix_struct *m_2 = get_matrix_struct(argv[2]);
      

  double *answer =
      (double *)aligned_alloc(64, m_1->rows * m_2->cols * sizeof(double));

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned int i = 0; i < m_1->rows; i++) {
    for (unsigned int j = 0; j < m_2->cols; j++) {
      __m512d sum_vec = _mm512_setzero_pd();

      for (unsigned int k = 0; k < m_1->cols; k += 8) {
        __m512d vec1 = _mm512_load_pd(&m_1->mat_data[i][k]);
        __m512d vec2 = _mm512_load_pd(&m_2->mat_data[k][j]);
        sum_vec = _mm512_fmadd_pd(vec1, vec2, sum_vec);
      }
      answer[i * m_2->cols + j] = _mm512_reduce_add_pd(sum_vec);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::micro> duration = end - start;
  std::cout << "size: " << m_1->rows << "x" << m_2->cols
            << " compute us: " << duration.count() << " comm us: 0"
            << std::endl;

  return 0;
}

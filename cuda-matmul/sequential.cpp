#include "matrix.hpp"
#include <chrono>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <matrix file> <matrix file>"
              << std::endl;
    return 1;
  }

  matrix_struct *m_1 = get_matrix_struct(argv[1]);
  matrix_struct *m_2 = get_matrix_struct(argv[2]);

  double *answer = (double *)malloc(m_1->rows * m_2->cols * sizeof(double));

  auto start = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < m_1->rows; i++) {
    for (unsigned int j = 0; j < m_2->cols; j++) {
      answer[i * m_2->cols + j] = 0;
      for (unsigned int k = 0; k < m_1->cols; k++) {
        answer[i * m_2->cols + j] += m_1->mat_data[i][k] * m_2->mat_data[k][j];
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::micro> duration = end - start;

  std::cout << "size: " << m_1->rows << "x" << m_2->cols
            << " compute us: " << duration.count() << " comm us: 0"
            << std::endl;

  return 0;
}

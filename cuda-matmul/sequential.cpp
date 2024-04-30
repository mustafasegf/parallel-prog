#include "matrix.cpp"
#include <chrono>
#include <iostream>
#include <stdint.h>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <matrix file> <matrix file>"
              << std::endl;
    return 1;
  }

  try {
    Matrix<int64_t> matrix1(argv[1]);
    Matrix<int64_t> matrix2(argv[2]);

    Matrix<int64_t> answer(matrix1.rows, matrix2.cols);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < matrix1.rows; i++) {
      for (size_t j = 0; j < matrix2.cols; j++) {
        answer[i][j] = 0;
        for (size_t k = 0; k < matrix1.cols; k++) {
          answer[i][j] += matrix1[i][k] * matrix2[k][j];
        }
      }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> duration = end - start;

    std::cout << "size: " << matrix1.rows << "x" << matrix2.cols
              << " compute us: " << duration.count() << " comm us: 0"
              << std::endl;

    std::cout << answer << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

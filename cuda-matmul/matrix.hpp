#ifndef MATRIX_HPP
#define MATRIX_HPP
#include "platform.h"

#pragma once

template <typename T> class AlignedAllocator {
public:
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using size_type = size_t;

  AlignedAllocator() noexcept {}

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

  pointer allocate(size_type num, const void * = 0) {
    pointer ret = (pointer)_mm_malloc(num * sizeof(T), 64);
    if (ret == nullptr) {
      throw std::bad_alloc();
    }
    return ret;
  }

  void deallocate(pointer p, size_type) { _mm_free(p); }

  bool operator==(const AlignedAllocator &) const { 
    return true; 
  }

  bool operator!=(const AlignedAllocator &other) const {
    return !(*this == other);
  }
};

template <typename T> class Matrix {
private:
public:
  std::vector<T, AlignedAllocator<T>> data;
  size_t rows, cols;
  Matrix(size_t rows, size_t cols)
      : data(rows * cols, 0, AlignedAllocator<T>()), rows(rows), cols(cols) {}

  explicit Matrix(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error(
          "Error: The file you entered could not be found.");
    }

    std::vector<T> loadedData;
    std::string line;
    size_t numCols = 0;

    while (getline(file, line)) {
      std::istringstream iss(line);
      T num;
      std::vector<T> row;
      while (iss >> num) {
        row.push_back(num);
      }
      if (!row.empty()) {
        if (numCols == 0) {
          numCols = row.size();
        } else if (row.size() != numCols) {
          throw std::runtime_error("Error: Irregular matrix shape detected.");
        }
        loadedData.insert(loadedData.end(), row.begin(), row.end());
      }
    }

    if (loadedData.empty() || numCols == 0) {
      throw std::runtime_error("Error: Empty matrix or incorrect data.");
    }

    rows = loadedData.size() / numCols;
    cols = numCols;
    data = std::vector<T, AlignedAllocator<T>>(loadedData.begin(),
                                               loadedData.end());
  }

  T &operator()(size_t row, size_t col) { return data[row * cols + col]; }

  const T &operator()(size_t row, size_t col) const {
    return data[row * cols + col];
  }

  T *operator[](size_t row) { return &data[row * cols]; }

  // get a pointer to the beginning of a matrix
  T *begin() { return &data[0]; }

  const T *operator[](size_t row) const { return &data[row * cols]; }

  // Overload the << operator to enable direct usage with std::cout
  friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix) {
    for (size_t i = 0; i < matrix.rows; ++i) {
      for (size_t j = 0; j < matrix.cols; ++j) {
        os << matrix.data[i * matrix.cols + j] << "\t";
      }
      os << "\n";
    }
    return os;
  }

  void print() const {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        std::cout << (*this)(i, j) << "\t";
      }
      std::cout << "\n";
    }
  }
};

#endif // MATRIX_HPP

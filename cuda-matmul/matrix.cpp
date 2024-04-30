#include "matrix.hpp"
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

matrix_struct *get_matrix_struct(char matrix[]) {
  matrix_struct *m = (matrix_struct *)aligned_alloc(64, sizeof(matrix_struct));
  m->rows = 0;
  m->cols = 0;
  FILE *myfile = fopen(matrix, "r");

  if (myfile == NULL) {
    printf("Error: The file you entered could not be found.\n");
    exit(EXIT_FAILURE);
  }

  int ch = 0;
  do {
    ch = fgetc(myfile);

    if (m->rows == 0 && ch == '\t')
      m->cols++;

    if (ch == '\n')
      m->rows++;

  } while (ch != EOF);

  m->cols++; // Adjust column count for the last column

  m->mat_data = (double **)malloc(m->rows * sizeof(double *));
  for (unsigned int i = 0; i < m->rows; ++i)
    m->mat_data[i] = (double *)aligned_alloc(64, m->cols * sizeof(double));

  rewind(myfile);
  unsigned int x, y;

  for (x = 0; x < m->rows; x++) {
    for (y = 0; y < m->cols; y++) {
      if (fscanf(myfile, "%lf", &m->mat_data[x][y]) != 1)
        break;
    }
  }

  fclose(myfile);
  return m;
}

void print_matrix(matrix_struct *matrix_to_print) {
  for (unsigned int i = 0; i < matrix_to_print->rows; i++) {
    for (unsigned int j = 0; j < matrix_to_print->cols; j++) {
      printf("%lf\t", matrix_to_print->mat_data[i][j]);
    }
    printf("\n");
  }
}

void free_matrix(matrix_struct *matrix_to_free) {
  for (unsigned int i = 0; i < matrix_to_free->rows; i++) {
    free(matrix_to_free->mat_data[i]);
  }
  free(matrix_to_free->mat_data);
  free(matrix_to_free);
}

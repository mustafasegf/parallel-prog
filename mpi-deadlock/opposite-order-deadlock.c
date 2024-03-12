#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int buf1 = 1, buf2 = 2;

  switch (rank) {
  case 0:
    MPI_Bcast(&buf1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&buf2, 1, MPI_INT, 1, MPI_COMM_WORLD);
    break;
  case 1:
    MPI_Bcast(&buf2, 1, MPI_INT, 1, MPI_COMM_WORLD);
    MPI_Bcast(&buf1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    break;
  default:
    printf("Rank %d: I'm not doing anything\n", rank);
  }

  MPI_Finalize();
  return 0;
}

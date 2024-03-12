#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int buf1 = 1, buf2 = 2;
  MPI_Status status;

  switch (rank) {
  case 0:
    MPI_Bcast(&buf1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Send(&buf2, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    break;
  case 1:
    MPI_Recv(&buf2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    // Barrier to ensure that the receive operation is completed before
    // proceeding to broadcast
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&buf1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    break;
  default:
    printf("Rank %d: I'm not doing anything\n", rank);
  }

  MPI_Finalize();
  return 0;
}

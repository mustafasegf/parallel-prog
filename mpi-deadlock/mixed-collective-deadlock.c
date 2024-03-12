#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  int size = 999999999;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // int sendbuf = 1, recvbuf = 2;
  // Allocate memory for send and receive buffers
  int *sendbuf = (int *)malloc(size * sizeof(int));
  int *recvbuf = (int *)malloc(size * sizeof(int));

  // Initialize send buffer
  for (int i = 0; i < size; i++) {
    sendbuf[i] = rank;
  }

  MPI_Status status;

  switch (rank) {
  case 0:
    // MPI_Bcast(&buf1, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Send(sendbuf, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(recvbuf, size, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
    break;
  case 1:
    MPI_Send(sendbuf, size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(recvbuf, size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    // MPI_Bcast(&sendbuf, size, MPI_INT, 0, MPI_COMM_WORLD);
    break;
  default:
    printf("Rank %d: I'm not doing anything\n", rank);
  }

  if (rank < 2) {
    printf("Rank %d received %d.\n", rank, *recvbuf);
  }

  MPI_Finalize();
  return 0;
}

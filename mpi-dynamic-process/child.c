#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  MPI_Comm parent_comm;
  MPI_Comm_get_parent(&parent_comm);

  if (parent_comm != MPI_COMM_NULL) {
    // This is the child process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int message;
    MPI_Recv(&message, 1, MPI_INT, 0, 0, parent_comm, MPI_STATUS_IGNORE);

    printf("Child (rank %d): received message %d from parent.\n", world_rank,
           message);
  }

  MPI_Finalize();
  return 0;
}

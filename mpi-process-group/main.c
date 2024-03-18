#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Determine color based on even/odd rank for splitting
  int color = rank % 4;

  // New communicator for the subgroup
  MPI_Comm new_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);

  // Each subgroup performs a simple collective operation
  int local_sum = rank + 1; // Just a simple operation for example
  int global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, new_comm);

  // Only rank 0 in each subgroup prints the result
  int new_rank, new_size;
  MPI_Comm_rank(new_comm, &new_rank);
  MPI_Comm_size(new_comm, &new_size);
  if (new_rank == 0) {
    printf("I am the master of my subgroup (original rank %2d), subgroup size: "
           "%d, global_sum: %5d, local_sum: %5d\n",
           rank, new_size, global_sum, local_sum);
  } else {
    printf("I am the slave  in my subgroup (original rank %2d), subgroup size: "
           "%d, global_sum: %5d, local_sum: %5d\n",
           rank, new_size, global_sum, local_sum);
  }

  // Free the new communicator
  MPI_Comm_free(&new_comm);

  MPI_Finalize();
  return 0;
}

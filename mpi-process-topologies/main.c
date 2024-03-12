#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Assuming we're using a 4x4 grid for simplicity; adjust for your use case
  int dims[2] = {4, 4};
  int periods[2] = {1, 1}; // Periodic in both dimensions for wrapping
  int reorder = 0;
  MPI_Comm grid_comm;

  // Create the Cartesian topology
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);

  // Get the Cartesian coordinates of the current process
  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);

  printf("Rank %d has coordinates (%d,%d)\n", rank, coords[0], coords[1]);

  // Determine neighbors
  int up, down, left, right;
  MPI_Cart_shift(grid_comm, 0, 1, &up, &down); // Shift in the first dimension
  MPI_Cart_shift(grid_comm, 1, 1, &left,
                 &right); // Shift in the second dimension

  printf("Rank %d's neighbors -> up: %d, down: %d, left: %d, right: %d\n", rank,
         up, down, left, right);

  // Cleanup
  MPI_Comm_free(&grid_comm);
  MPI_Finalize();
  return 0;
}

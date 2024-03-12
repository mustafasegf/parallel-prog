#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int buf1 = 1, buf2 = 2;

    // Adding MPI_Barrier to ensure synchronization before starting broadcast operations
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        MPI_Bcast(&buf1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Bcast(&buf2, 1, MPI_INT, 1, MPI_COMM_WORLD);
    }

    // Synchronize again to ensure all processes have completed their first broadcast
    MPI_Barrier(MPI_COMM_WORLD);

    // Swap the order of broadcasts with a barrier between them to prevent deadlock
    if (rank == 0) {
        MPI_Bcast(&buf2, 1, MPI_INT, 1, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Bcast(&buf1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}


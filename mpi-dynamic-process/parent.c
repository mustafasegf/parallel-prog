#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm parent_comm, intercomm;

  MPI_Comm_get_parent(&parent_comm);

  if (parent_comm == MPI_COMM_NULL) {
    // This is the parent process
    char *child_program = "./child";
    MPI_Info info;
    MPI_Info_create(&info);
    // MPI_Info_set(info, "key", "value"); // Example setting if needed

    int errcodes[1];
    MPI_Comm_spawn(child_program, MPI_ARGV_NULL, 1, info, 0, MPI_COMM_WORLD,
                   &intercomm, errcodes);

    // Example: send a message to the child process
    int message = 123;
    MPI_Send(&message, 1, MPI_INT, 0, 0, intercomm);

    printf("Parent (rank %d): sent message %d to child.\n", world_rank,
           message);
  }

  MPI_Finalize();
  return 0;
}

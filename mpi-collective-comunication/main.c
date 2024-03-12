#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void broadcastExample() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int data;
  if (rank == 0) {
    // Root process initializes data
    data = 100;
  }

  // Broadcast data from root (rank 0) to all processes
  MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Now all processes have the data value
  printf("Rank %d received data: %d\n", rank, data);
}

void reductionExample() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int value = rank + 1; // Each process contributes a unique value
  int result;

  // Sum all values, result stored in rank 0
  MPI_Reduce(&value, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("The sum is: %d\n", result);
  }
}

void gatherExample() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Each process creates its own data
  int sendData = rank + 1;
  int *recvData = NULL;

  if (rank == 0) {
    recvData = (int *)malloc(size * sizeof(int));
  }

  MPI_Gather(&sendData, 1, MPI_INT, recvData, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < size; i++) {
      printf("%d ", recvData[i]);
    }
    printf("\n");
    free(recvData);
  }
}

void scatterExample() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int *sendData = NULL;
  int recvData;

  if (rank == 0) {
    sendData = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
      sendData[i] = i + 1; // Initialize array to be scattered
    }
  }

  MPI_Scatter(sendData, 1, MPI_INT, &recvData, 1, MPI_INT, 0, MPI_COMM_WORLD);

  printf("Rank %d received %d\n", rank, recvData);

  if (rank == 0) {
    free(sendData);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  broadcastExample();
  reductionExample();
  gatherExample();
  scatterExample();

  MPI_Finalize();
  return 0;
}

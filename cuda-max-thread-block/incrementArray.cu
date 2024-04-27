#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>

void incrementArrayOnHost(float *a, long long N) {
  for (long long i = 0; i < N; i++) {
    a[i] = a[i] + 1.f;
  }
}
__global__ void incrementArrayOnDevice(float *a, long long N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    a[idx] = a[idx] + 1.f;
  }
}

int main(int argc, char **argv) {
  float *a_h, *b_h; // pointers to host memory
  float *a_d;       // pointer to device memory
  long long i, N = 10;

  if (argc > 1) {
    N = atoll(argv[1]);
  }

  size_t size = N * sizeof(float);
  // allocate arrays on host
  a_h = (float *)malloc(size);
  b_h = (float *)malloc(size);
  // allocate array on device
  cudaMalloc((void **)&a_d, size);
  // initialization of host data
  for (i = 0; i < N; i++) {
    a_h[i] = (float)i;
  }
  // copy data from host to device
  cudaMemcpy(a_d, a_h, sizeof(float) * N, cudaMemcpyHostToDevice);
  // do calculation on device:
  // Part 1 of 2. Compute execution configuration
  long long blockSize = 4;
  if (argc > 2) {
    blockSize = atoll(argv[2]);
  }

  // do calculation on host
  incrementArrayOnHost(a_h, N);

  long long nBlocks = N / blockSize + (N % blockSize == 0 ? 0 : 1);

  dim3 dimBlock(nBlocks);

  std::cout << "nBlocks: " << nBlocks << std::endl;
  std::cout << "x: " << dimBlock.x << " y: " << dimBlock.y
            << " z: " << dimBlock.z << std::endl;

  // Part 2 of 2. Call incrementArrayOnDevice kernel
  incrementArrayOnDevice<<<dimBlock, blockSize>>>(a_d, N);
  // Retrieve result from device and store in b_h
  cudaMemcpy(b_h, a_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
  // check results
  for (i = 0; i < N; i++) {
    assert(a_h[i] == b_h[i]);
  }

  // std::cout << "Results are correct!" << std::endl;

  // for (i = 0; i < N; i++) {
  //   std::cout << b_h[i] << " ";
  // }

  // cleanup
  free(a_h);
  free(b_h);
  cudaFree(a_d);
}

#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>

void incrementArrayOnHost(float *a, int64_t N) {
  for (size_t i = 0; i < N; i++) {
    a[i] = a[i] + 1.f;
  }
}
__global__ void incrementArrayOnDevice(float *a, int64_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    a[idx] = a[idx] + 1.f;
  }
}

int main(int argc, char **argv) {
  int64_t gridDim = 3; // Number of blocks in the grid
  if (argc > 1) {
    gridDim = atoll(argv[1]);
  }

  int64_t blockDim = 5; // Number of threads in a block
  if (argc > 2) {
    blockDim = atoll(argv[2]);
  }

  float *a_host, *b_host; // pointers to host memory
  float *a_device;        // pointer to device memory

  int64_t N = gridDim * blockDim;
  int64_t size = N * sizeof(float);

  dim3 grid(gridDim);
  dim3 blocks(blockDim);

  a_host = (float *)malloc(size);
  b_host = (float *)malloc(size);
  cudaMalloc((void **)&a_device, size);

  for (size_t i = 0; i < N; i++) {
    a_host[i] = (float)i;
  }
  cudaMemcpy(a_device, a_host, sizeof(float) * N, cudaMemcpyHostToDevice);

  incrementArrayOnHost(a_host, N);

  std::cout << "grid x: " << grid.x << " y: " << grid.y << " z: " << grid.z
            << std::endl;
  std::cout << "block x: " << blocks.x << " y: " << blocks.y
            << " z: " << blocks.z << std::endl;

  incrementArrayOnDevice<<<grid, blocks>>>(a_device, N);
  cudaMemcpy(b_host, a_device, sizeof(float) * N, cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < N; i++) {
    assert(a_host[i] == b_host[i]);
  }

  // std::cout << "Results are correct!" << std::endl;

  // for (size_t i = 0; i < N; i++) {
  //   std::cout << b_host[i] << " ";
  // }

  // cleanup
  free(a_host);
  free(b_host);
  cudaFree(a_device);
}

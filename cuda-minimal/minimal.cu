#include <iostream>

__global__ void kernelIdx(int *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[idx] = idx;
}
__global__ void kernelThreadIdx(int *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[idx] = threadIdx.x;
}
__global__ void kernelBlockIdx(int *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[idx] = blockIdx.x;
}
__global__ void kernelBlockDim(int *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[idx] = blockDim.x;
}
__global__ void kernelGridDim(int *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[idx] = gridDim.x;
}
__global__ void kernelStatic(int *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[idx] = 69;
}

int main(int argc, char *argv[]) {
  int blockDim = 5; // Number of threads in a block
  if (argc > 1) {
    blockDim = atoi(argv[1]);
  }

  int gridDim = 3; // Number of blocks in the grid
  if (argc > 2) {
    gridDim = atoi(argv[2]);
  }

  int *a_device, *a_host;
  int N = gridDim * blockDim;
  size_t size = N * sizeof(int);

  a_host = (int *)malloc(size);
  cudaMalloc(&a_device, size);

  dim3 grid(gridDim);
  dim3 blocks(blockDim);

  kernelIdx<<<grid, blocks>>>(a_device);
  cudaMemcpy(a_host, a_device, size, cudaMemcpyDeviceToHost);

  std::cout << "kernelIdx" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << a_host[i] << " ";
  }
  std::cout << std::endl;

  kernelThreadIdx<<<grid, blocks>>>(a_device);
  cudaMemcpy(a_host, a_device, size, cudaMemcpyDeviceToHost);
  std::cout << "kernelThreadIdx" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << a_host[i] << " ";
  }
  std::cout << std::endl;

  kernelBlockIdx<<<grid, blocks>>>(a_device);
  cudaMemcpy(a_host, a_device, size, cudaMemcpyDeviceToHost);
  std::cout << "kernelBlockIdx" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << a_host[i] << " ";
  }
  std::cout << std::endl;

  kernelBlockDim<<<grid, blocks>>>(a_device);
  cudaMemcpy(a_host, a_device, size, cudaMemcpyDeviceToHost);
  std::cout << "kernelBlockDim" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << a_host[i] << " ";
  }
  std::cout << std::endl;

  kernelGridDim<<<grid, blocks>>>(a_device);
  cudaMemcpy(a_host, a_device, size, cudaMemcpyDeviceToHost);
  std::cout << "kernelGridDim" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << a_host[i] << " ";
  }
  std::cout << std::endl;

  kernelStatic<<<grid, blocks>>>(a_device);
  cudaMemcpy(a_host, a_device, size, cudaMemcpyDeviceToHost);
  std::cout << "kernelStatic" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << a_host[i] << " ";
  }
  std::cout << std::endl;

  cudaFree(a_device);
  free(a_host);

  return 0;
}

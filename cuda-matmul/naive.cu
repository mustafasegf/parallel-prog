#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMulKernel(double *A, double *B, double *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    double sum = 0.0;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

int main() {
  int N = 512;
  size_t bytes = N * N * sizeof(double);

  double *h_A = (double *)malloc(bytes);
  double *h_B = (double *)malloc(bytes);
  double *h_C = (double *)malloc(bytes);

  // Initialize matrices
  for (int i = 0; i < N * N; i++) {
    h_A[i] = 1.0;
    h_B[i] = 2.0;
    h_C[i] = 0.0;
  }

  double *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);

  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}

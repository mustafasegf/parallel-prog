#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMulSharedKernel(double *A, double *B, double *C, int N) {
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int aBegin = N * blockDim.y * by;
  int aEnd = aBegin + N - 1;
  int aStep = blockDim.x;
  int bBegin = blockDim.x * bx;
  int bStep = blockDim.x * N;
  double sum = 0.0;

  __shared__ double As[16][16];
  __shared__ double Bs[16][16];

  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    As[ty][tx] = A[a + N * ty + tx];
    Bs[ty][tx] = B[b + N * ty + tx];
    __syncthreads();

    for (int k = 0; k < blockDim.x; ++k) {
      sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }
  int c = N * blockDim.y * by + blockDim.x * bx;
  C[c + N * ty + tx] = sum;
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

  matrixMulSharedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}

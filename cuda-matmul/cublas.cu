#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// Helper function to fill the matrix with double precision values
void fillMatrix(double *mat, int n, double val) {
  for (int i = 0; i < n * n; ++i) {
    mat[i] = val;
  }
}

int main() {
  int N = 512;
  size_t bytes = N * N * sizeof(double);

  // Allocate host memory
  double *h_A = (double *)malloc(bytes);
  double *h_B = (double *)malloc(bytes);
  double *h_C = (double *)malloc(bytes);

  // Initialize matrices
  fillMatrix(h_A, N, 1.0);
  fillMatrix(h_B, N, 2.0);
  fillMatrix(h_C, N, 0.0);

  // Allocate device memory
  double *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);

  // Copy matrices from the host to the device
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Define scalar values for axpy operations
  const double alpha = 1.0;
  const double beta = 0.0;

  // Perform matrix multiplication: C = alpha * A * B + beta * C
  // Note: cuBLAS assumes column-major storage, whereas C/C++ uses row-major.
  // Hence, we invert the operation order.
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N,
              &beta, d_C, N);

  // Copy the result back to host
  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

  // Print the result
  std::cout << "Result: " << h_C[0] << std::endl;

  // Clean up resources
  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}

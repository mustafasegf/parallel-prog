# Compiler
NVCC = nvcc
CC = g++

# Compiler flags
NVCCFLAGS = -std=c++11 -O3 
CFLAGS = -std=c++11 -O3 -pedantic -Wall -Wextra -Wconversion -funroll-loops -ffast-math -march=native -mavx512dq

# Libraries
NVCCLIBRARIES = -lcublas
CLIBRARIES = matrix.cpp

# Build targets
TARGETS = cublas naive shared sequential avx

# Source files
CUBLAS_SRC = cublas.cu
MATMUL_SRC = matmul.cu
SEQUENTIAL_SRC = sequential.cpp

all: $(TARGETS)

naive: $(MATMUL_SRC)
	$(NVCC) $(NVCCFLAGS) $(MATMUL_SRC) -o $@ 

shared: $(MATMUL_SRC)
	$(NVCC) $(NVCCFLAGS) $(MATMUL_SRC) -o $@ -DSHARED

cublas: $(CUBLAS_SRC)
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_SRC) -o $@ $(NVCCLIBRARIES)

sequential: $(SEQUENTIAL_SRC) $(CLIBRARIES)
	$(CC) $(CFLAGS) $(SEQUENTIAL_SRC) -o $@ $(CLIBRARIES) -DTILING

avx: $(SEQUENTIAL_SRC) $(CLIBRARIES)
	$(CC) $(CFLAGS) $(SEQUENTIAL_SRC) -o $@ $(CLIBRARIES) -DAVX  -mavx512dq -mavx512vl -mavx512bw -mavx512f

clean:
	rm -f $(TARGETS)

.PHONY: all clean


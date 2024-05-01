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
NAIVE_SRC = naive.cu
SHARED_SRC = shared.cu
SEQUENTIAL_SRC = sequential.cpp

all: $(TARGETS)

cublas: $(CUBLAS_SRC)
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_SRC) -o $@ $(NVCCLIBRARIES)

naive: $(NAIVE_SRC)
	$(NVCC) $(NVCCFLAGS) $(NAIVE_SRC) -o $@

shared: $(SHARED_SRC)
	$(NVCC) $(NVCCFLAGS) $(SHARED_SRC) -o $@

sequential: $(SEQUENTIAL_SRC) $(CLIBRARIES)
	$(CC) $(CFLAGS) $(SEQUENTIAL_SRC) -o $@ $(CLIBRARIES) -DTILING

avx: $(SEQUENTIAL_SRC) $(CLIBRARIES)
	$(CC) $(CFLAGS) $(SEQUENTIAL_SRC) -o $@ $(CLIBRARIES) -DAVX  -mavx512dq -mavx512vl -mavx512bw -mavx512f

clean:
	rm -f $(TARGETS)

.PHONY: all clean


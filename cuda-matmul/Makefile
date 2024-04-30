# Compiler
NVCC = nvcc
CC = g++

# Compiler flags
NVCCFLAGS = -std=c++11 -O2 -pedantic -Wall -Wextra -Wconversion
CFLAGS = -std=c++11 -O2 -pedantic -Wall -Wextra -Wconversion

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
AVX_SRC = avx.cpp

all: $(TARGETS)

cublas: $(CUBLAS_SRC)
	$(NVCC) $(NVCCFLAGS) $(CUBLAS_SRC) -o $@ $(NVCCLIBRARIES)

naive: $(NAIVE_SRC)
	$(NVCC) $(NVCCFLAGS) $(NAIVE_SRC) -o $@

shared: $(SHARED_SRC)
	$(NVCC) $(NVCCFLAGS) $(SHARED_SRC) -o $@

sequential: $(SEQUENTIAL_SRC) $(CLIBRARIES)
	$(CC) $(CFLAGS) $(SEQUENTIAL_SRC) -o $@ $(CLIBRARIES)

avx: $(AVX_SRC) $(CLIBRARIES)
	$(CC) $(CFLAGS) $(AVX_SRC) -o $@ $(CLIBRARIES)  -mavx512f

clean:
	rm -f $(TARGETS)

.PHONY: all clean


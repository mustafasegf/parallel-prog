# Compiler
NVCC = nvcc
CC = g++

# Compiler flags
NVCCFLAGS = -std=c++11 -O3 
CFLAGS = -std=c++11 -O3 -pedantic -Wall -Wextra -Wconversion -funroll-loops -ffast-math -march=native

# Libraries
NVCCLIBRARIES = -lcublas
CLIBRARIES = 

# Build targets
TARGETS = cublas naive shared sequential tiling

# check if avx2 is supported
ifneq ($(shell grep -c avx2 /proc/cpuinfo),0)
TARGETS += avx2
endif

# check if avx512 is supported
ifneq ($(shell grep -c avx512 /proc/cpuinfo),0)
TARGETS += avx512
endif

# Source files
MATMUL_SRC = matmul.cu
SEQUENTIAL_SRC = sequential.cpp
HEADERS = matrix.hpp

all: $(TARGETS)

naive: $(MATMUL_SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(MATMUL_SRC) -o $@ 

shared: $(MATMUL_SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(MATMUL_SRC) -o $@ -DSHARED

cublas: $(MATMUL_SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(MATMUL_SRC) -o $@ -lcublas -DCUBLAS

sequential: $(SEQUENTIAL_SRC) $(CLIBRARIES) $(HEADERS)
	$(CC) $(CFLAGS) $(SEQUENTIAL_SRC) -o $@ $(CLIBRARIES)

tiling: $(SEQUENTIAL_SRC) $(CLIBRARIES) $(HEADERS) 
	$(CC) $(CFLAGS) $(SEQUENTIAL_SRC) -o $@ $(CLIBRARIES) -DTILING

avx2: $(SEQUENTIAL_SRC) $(CLIBRARIES) $(HEADERS)
	$(CC) $(CFLAGS) $(SEQUENTIAL_SRC) -o $@ $(CLIBRARIES) -DAVX2 -mavx2 -mfma

avx512: $(SEQUENTIAL_SRC) $(CLIBRARIES) $(HEADERS)
	$(CC) $(CFLAGS) $(SEQUENTIAL_SRC) -o $@ $(CLIBRARIES) -DAVX512  -mavx512dq -mavx512vl -mavx512bw -mavx512f

clean:
	rm -f $(TARGETS)

.PHONY: all clean


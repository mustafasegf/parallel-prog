# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -std=c++11 -O2

# Build target
TARGET = minimal

# Source file
SOURCE = minimal.cu

# Build rule for the main target
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Clean up
clean:
	rm -f $(TARGET)

# Run the program with a default block size
run: $(TARGET)
	./$(TARGET) 10

# Phony targets
.PHONY: all clean run


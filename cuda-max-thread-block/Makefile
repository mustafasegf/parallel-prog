NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3
TARGET = incrementArray
SOURCE = incrementArray.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean

#ifndef PLATFORM_H
#define PLATFORM_H

#ifndef __cplusplus
#error A C++ compiler is required!
#endif

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#include <immintrin.h>
#else
#include <cstdlib>
static inline void *_emulate_mm_malloc(size_t size, size_t align) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, align, size) != 0) {
    throw std::bad_alloc();
  }
  return ptr;
}
#define _mm_malloc _emulate_mm_malloc
#define _mm_free free
#endif 

#endif // PLATFORM_H

#pragma once

#include <stddef.h>

// length of a single vector register (in bytes)
static const unsigned int VLEN = 512 / 8;

typedef double f64;
typedef unsigned long long u64;

#ifdef __cplusplus
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

void *malloc_aligned(size_t size, size_t align) __attribute__((malloc, alloc_size(1)));
static inline int min_int(int a, int b) { return a < b ? a : b; }

#ifdef __cplusplus
}
#endif
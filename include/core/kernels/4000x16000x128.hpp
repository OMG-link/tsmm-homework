#pragma once
#include <cstdio>

#include "core/optimized.hpp"

class MatMul4000x16000x128 : public MatMulOptimizedKernel {
  public:
    bool match(int M, int N, int K) const override { return M == 4000 && K == 16000 && N == 128; }

    void compute(const f64 *A, const f64 *B, f64 *C, int M, int N, int K) const override {
        // TODO
        puts("Using optimized kernel");
    }
};
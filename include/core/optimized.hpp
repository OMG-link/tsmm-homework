#pragma once
#include "matrix.hpp"

class MatMulOptimizedKernel {
  public:
    virtual ~MatMulOptimizedKernel() = default;
    virtual bool match(int M, int N, int K) const = 0;
    virtual void compute(const f64 *A, const f64 *B, f64 *C, int M, int N, int K) const = 0;
};
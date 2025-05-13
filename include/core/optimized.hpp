#pragma once
#include "matrix.hpp"

class MatMulOptimizedKernel {
  public:
    virtual ~MatMulOptimizedKernel() = default;
    virtual bool match(int m, int k, int n) const = 0;
    virtual void compute(const f64 *lhs, const f64 *rhs, f64 *dst, int m, int k, int n) const = 0;
};
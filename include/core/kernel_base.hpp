#pragma once
#include "matrix.hpp"

class MatmulKernelBase {
  public:
    virtual ~MatmulKernelBase() = default;
    virtual bool match(int m, int k, int n) const = 0;
    virtual void compute(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n) const = 0;
};
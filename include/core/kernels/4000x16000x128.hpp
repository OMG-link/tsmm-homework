#pragma once
#include <cassert>
#include <cstdio>

#include "core/optimized.hpp"

class MatMul4000x16000x128 : public MatMulOptimizedKernel {
  public:
    bool match(int m, int k, int n) const override { return m == 4000 && k == 16000 && n == 128; }

    void compute(const f64 *lhs, const f64 *rhs, f64 *dst, int m, int n, int k) const override {
        assert(match(m, n, k));
        // TODO
    }
};
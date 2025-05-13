#pragma once
#include <cassert>
#include <cstdlib>

#include "core/kernel_base.hpp"
#include "core/matmul.h"

static const int M_BLK = 64;
static const int K_BLK = 64;
static const int N_BLK = 64;

class MatMul4000x16000x128 : public MatmulKernelBase {
  public:
    bool match(int m, int k, int n) const override { return m == 4000 && k == 16000 && n == 128; }

    void compute(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n) const override {
        assert(match(m, k, n));
        matmul_block(dst, lhs, rhs, m, k, n, M_BLK, K_BLK, N_BLK);
    }
};
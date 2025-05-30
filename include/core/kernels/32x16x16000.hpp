#pragma once
#include <cassert>
#include <cstdlib>

#include "core/kernel_base.hpp"

class MatMul32x16x16000 : public MatmulKernelBase {
  public:
    bool match(int m, int k, int n) const override { return m == 32 && k == 16 && n == 16000; }

    void compute(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k,
                 int n) const override;
};
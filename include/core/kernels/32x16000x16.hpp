#pragma once
#include <cassert>
#include <cstdlib>

#include "core/kernel_base.hpp"

class MatMul32x16000x16 : public MatmulKernelBase {
  public:
    bool match(int m, int k, int n) const override { return m == 32 && k == 16000 && n == 16; }

    void compute(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k,
                 int n) const override;
};
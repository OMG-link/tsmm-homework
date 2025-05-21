#pragma once
#include <cassert>
#include <cstdlib>

#include "core/kernel_base.hpp"

class MatMul4000x128x16000 : public MatmulKernelBase {
  public:
    bool match(int m, int k, int n) const override { return m == 4000 && k == 128 && n == 16000; }

    void compute(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k,
                 int n) const override;
};
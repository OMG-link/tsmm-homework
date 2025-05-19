#pragma once
#include <cassert>
#include <cstdlib>

#include "core/kernel_base.hpp"

class MatMul144x144x144 : public MatmulKernelBase {
  public:
    bool match(int m, int k, int n) const override { return m == 144 && k == 144 && n == 144; }

    void compute(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k,
                 int n) const override;
};
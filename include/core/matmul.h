#pragma once

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void matmul_block(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k, int n,
                  const int M_BLK, const int K_BLK, const int N_BLK);
void matmul_submat(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k, int n,
                   int dst_line_stride, int zero_dst);

#ifdef __cplusplus
}
#endif

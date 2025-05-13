#pragma once

#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

void matmul_block(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n, const int M_BLK, const int K_BLK,
                  const int N_BLK);
void matmul_submat(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n, int dst_line_stride);

#ifdef __cplusplus
}
#endif

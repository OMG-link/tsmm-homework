#pragma once

#include "utils.h"

#define outer_product_kernel(m_block, n_block)                                                                         \
    f64 out[m_block * n_block];                                                                                        \
    for (int i = 0; i < m_block; i++) {                                                                                \
        for (int j = 0; j < n_block; j++) {                                                                            \
            out[i * n_block + j] = dst[(m_idx + i) * dst_line_stride + (n_idx + j)];                                   \
        }                                                                                                              \
    }                                                                                                                  \
    for (int k_idx = 0; k_idx < k; k_idx++) {                                                                          \
        for (int i = 0; i < m_block; i++) {                                                                            \
            for (int j = 0; j < n_block; j++) {                                                                        \
                out[i * n_block + j] += lhs[m_idx * k + k_idx * m_block + i] * rhs[n_idx * k + k_idx * n_block + j];   \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    for (int i = 0; i < m_block; i++) {                                                                                \
        for (int j = 0; j < n_block; j++) {                                                                            \
            dst[(m_idx + i) * dst_line_stride + (n_idx + j)] = out[i * n_block + j];                                   \
        }                                                                                                              \
    }

#ifdef __cplusplus
extern "C" {
#endif

void matmul_block(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n, const int M_BLK, const int K_BLK,
                  const int N_BLK);
void matmul_submat(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n, int dst_line_stride);

#ifdef __cplusplus
}
#endif

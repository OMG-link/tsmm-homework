#include "core/matmul.h"

#include <stdlib.h>
#include <string.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "core/pack.h"

static inline int min(int a, int b) { return a < b ? a : b; }

// Compute dst = lhs * rhs using blocking strategy.
void matmul_block(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n, const int M_BLK, const int K_BLK,
                  const int N_BLK) {
    // pack
    f64 *lhs_packed = (f64 *)malloc(m * k * sizeof(f64));
    f64 *rhs_packed = (f64 *)malloc(k * n * sizeof(f64));
    pack_matrix_lhs(lhs_packed, lhs, m, k, M_BLK, K_BLK);
    pack_matrix_rhs(rhs_packed, rhs, k, n, K_BLK, N_BLK);

    // clean dst
    memset(dst, 0, m * n * sizeof(f64));

    // do mat mul
    for (int m_idx = 0, m_block; m_idx < m; m_idx += m_block) {
        m_block = min(M_BLK, m - m_idx);
        for (int n_idx = 0, n_block; n_idx < n; n_idx += n_block) {
            n_block = min(N_BLK, n - n_idx);
            for (int k_idx = 0, k_block; k_idx < k; k_idx += k_block) {
                k_block = min(K_BLK, k - k_idx);
                const f64 *lhs_submat = lhs_packed + m_idx * k + k_idx * m_block;
                const f64 *rhs_submat = rhs_packed + n_idx * k + k_idx * n_block;
                f64 *dst_submat = dst + m_idx * n + n_idx;
                matmul_submat(dst_submat, lhs_submat, rhs_submat, m_block, k_block, n_block, n);
            }
        }
    }

    free(lhs_packed);
    free(rhs_packed);
}

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

void matmul_submat(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n, int dst_line_stride) {
    int m_idx;
    for (m_idx = 0; m_idx + DST_M_BLK <= m; m_idx += DST_M_BLK) {
        int n_idx;
        for (n_idx = 0; n_idx + DST_N_BLK <= n; n_idx += DST_N_BLK) {
#ifdef __AVX512F__
            __m512d out[DST_M_BLK];
            for (int i = 0; i < DST_M_BLK; i++) {
                out[i] = _mm512_loadu_pd(&dst[(m_idx + i) * dst_line_stride + n_idx]);
            }
            for (int k_idx = 0; k_idx < k; k_idx++) {
                for (int i = 0; i < DST_M_BLK; i++) {
                    double lhs_val = lhs[m_idx * k + k_idx * DST_M_BLK + i];
                    __m512d lhs_broadcast = _mm512_set1_pd(lhs_val);
                    const double *rhs_vec_ptr = &rhs[n_idx * k + k_idx * DST_N_BLK];
                    __m512d rhs_vec = _mm512_loadu_pd(rhs_vec_ptr);
                    out[i] = _mm512_fmadd_pd(lhs_broadcast, rhs_vec, out[i]);
                }
            }
            for (int i = 0; i < DST_M_BLK; i++) {
                _mm512_storeu_pd(&dst[(m_idx + i) * dst_line_stride + n_idx], out[i]);
            }
#else
            outer_product_kernel(DST_M_BLK, DST_N_BLK);
#endif
        }
        {
            const int n_block = n - n_idx;
            outer_product_kernel(DST_M_BLK, n_block);
        }
    }
    {
        const int m_block = m - m_idx;
        int n_idx;
        for (n_idx = 0; n_idx + DST_N_BLK <= n; n_idx += DST_N_BLK) {
            outer_product_kernel(m_block, DST_N_BLK);
        }
        {
            const int n_block = n - n_idx;
            outer_product_kernel(m_block, n_block);
        }
    }
}
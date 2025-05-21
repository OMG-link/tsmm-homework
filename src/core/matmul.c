#include "core/matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/pack.h"

static const int OPK_M_BLK = 8;
static const int OPK_N_BLK = 24;

// Compute dst = lhs * rhs using blocking strategy.
void matmul_block(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k, int n,
                  const int M_BLK, const int K_BLK, const int N_BLK) {
    // pack
    f64 *lhs_packed = (f64 *)malloc_aligned((m * k) * sizeof(f64), 128);
    f64 *rhs_packed = (f64 *)malloc_aligned((k * n) * sizeof(f64), 128);
    pack_matrix_lhs(lhs_packed, lhs, m, k, M_BLK, K_BLK, OPK_M_BLK);
    pack_matrix_rhs(rhs_packed, rhs, k, n, K_BLK, N_BLK, OPK_N_BLK);

    // clean dst
    memset(dst, 0, m * n * sizeof(f64));

    // do mat mul
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int m_idx = 0; m_idx < m; m_idx += M_BLK) {
        for (int n_idx = 0; n_idx < n; n_idx += N_BLK) {
            int m_block = min_int(M_BLK, m - m_idx);
            int n_block = min_int(N_BLK, n - n_idx);
            for (int k_idx = 0, k_block; k_idx < k; k_idx += k_block) {
                k_block = min_int(K_BLK, k - k_idx);
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

#ifdef __AVX512F__
#define load_out(i, j) out##i##j = _mm512_loadu_pd(dst_ptr + i * dst_line_stride + j * 8)
#define fma(i, j) out##i##j = _mm512_fmadd_pd(lhs_vbc##i, rhs_vec##j, out##i##j)
#define store_out(i, j) _mm512_storeu_pd(dst_ptr + i * dst_line_stride + j * 8, out##i##j)
#endif

void matmul_submat(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k, int n,
                   int dst_line_stride) {
    int m_idx;
    for (m_idx = 0; m_idx + OPK_M_BLK <= m; m_idx += OPK_M_BLK) {
        int n_idx = 0;
        for (; n_idx + OPK_N_BLK <= n; n_idx += OPK_N_BLK) {
#ifdef __AVX512F__
            const f64 *lhs_val_ptr = lhs + m_idx * k;
            const f64 *rhs_vec_ptr = rhs + n_idx * k;
            f64 *dst_ptr = dst + m_idx * dst_line_stride + n_idx;
            __m512d out00, out10, out20, out30, out40, out50, out60, out70;
            __m512d out01, out11, out21, out31, out41, out51, out61, out71;
            __m512d out02, out12, out22, out32, out42, out52, out62, out72;
            load_out(0, 0), load_out(1, 0), load_out(2, 0), load_out(3, 0);
            load_out(4, 0), load_out(5, 0), load_out(6, 0), load_out(7, 0);
            load_out(0, 1), load_out(1, 1), load_out(2, 1), load_out(3, 1);
            load_out(4, 1), load_out(5, 1), load_out(6, 1), load_out(7, 1);
            load_out(0, 2), load_out(1, 2), load_out(2, 2), load_out(3, 2);
            load_out(4, 2), load_out(5, 2), load_out(6, 2), load_out(7, 2);
            for (int k_idx = 0; k_idx < k; k_idx++) {
                const int PREFETCH_ITER = 8;
                const int RHS_PREFETCH_DIST = PREFETCH_ITER * OPK_N_BLK;
                const int LHS_PREFETCH_DIST = PREFETCH_ITER * OPK_M_BLK;
                __m512d rhs_vec0 = _mm512_loadu_pd(rhs_vec_ptr + 0);
                __m512d rhs_vec1 = _mm512_loadu_pd(rhs_vec_ptr + 8);
                __m512d rhs_vec2 = _mm512_loadu_pd(rhs_vec_ptr + 16);
                __m512d lhs_vbc0 = _mm512_set1_pd(lhs_val_ptr[0]);
                __m512d lhs_vbc1 = _mm512_set1_pd(lhs_val_ptr[1]);
                __m512d lhs_vbc2 = _mm512_set1_pd(lhs_val_ptr[2]);
                __m512d lhs_vbc3 = _mm512_set1_pd(lhs_val_ptr[3]);
                fma(0, 0), fma(1, 0), fma(2, 0), fma(3, 0);
                fma(0, 1), fma(1, 1), fma(2, 1), fma(3, 1);
                fma(0, 2), fma(1, 2), fma(2, 2), fma(3, 2);
                __m512d lhs_vbc4 = _mm512_set1_pd(lhs_val_ptr[4]);
                __m512d lhs_vbc5 = _mm512_set1_pd(lhs_val_ptr[5]);
                __m512d lhs_vbc6 = _mm512_set1_pd(lhs_val_ptr[6]);
                __m512d lhs_vbc7 = _mm512_set1_pd(lhs_val_ptr[7]);
                fma(4, 0), fma(5, 0), fma(6, 0), fma(7, 0);
                fma(4, 1), fma(5, 1), fma(6, 1), fma(7, 1);
                fma(4, 2), fma(5, 2), fma(6, 2), fma(7, 2);
                _mm_prefetch(rhs_vec_ptr + RHS_PREFETCH_DIST + 0, _MM_HINT_T0);
                _mm_prefetch(rhs_vec_ptr + RHS_PREFETCH_DIST + 8, _MM_HINT_T0);
                _mm_prefetch(rhs_vec_ptr + RHS_PREFETCH_DIST + 16, _MM_HINT_T0);
                _mm_prefetch(lhs_val_ptr + LHS_PREFETCH_DIST + 0, _MM_HINT_T0);
                lhs_val_ptr += OPK_M_BLK;
                rhs_vec_ptr += OPK_N_BLK;
            }
            store_out(0, 0), store_out(1, 0), store_out(2, 0), store_out(3, 0);
            store_out(4, 0), store_out(5, 0), store_out(6, 0), store_out(7, 0);
            store_out(0, 1), store_out(1, 1), store_out(2, 1), store_out(3, 1);
            store_out(4, 1), store_out(5, 1), store_out(6, 1), store_out(7, 1);
            store_out(0, 2), store_out(1, 2), store_out(2, 2), store_out(3, 2);
            store_out(4, 2), store_out(5, 2), store_out(6, 2), store_out(7, 2);
#else
            outer_product_kernel(OPK_M_BLK, OPK_N_BLK);
#endif
        }
        if (n - n_idx > 0) {
            const int n_block = n - n_idx;
            outer_product_kernel(OPK_M_BLK, n_block);
        }
    }
    if (m - m_idx > 0) {
        const int m_block = m - m_idx;
        int n_idx;
        for (n_idx = 0; n_idx + OPK_N_BLK <= n; n_idx += OPK_N_BLK) {
            outer_product_kernel(m_block, OPK_N_BLK);
        }
        if (n - n_idx > 0) {
            const int n_block = n - n_idx;
            outer_product_kernel(m_block, n_block);
        }
    }
}

#ifdef __AVX512F__
#undef load_out
#undef fma
#undef store_out
#endif
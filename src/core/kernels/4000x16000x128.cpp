#include "core/kernels/4000x16000x128.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "core/pack.h"

static const int M = 4000;
static const int K = 16000;
static const int N = 128;

static const int M_BLK = 128;
static const int K_BLK = 256;
static const int N_BLK = 128;

static const int PREFETCH_ITER = 4;
static const int RHS_PREFETCH_DIST = PREFETCH_ITER * DST_N_BLK;
static const int LHS_PREFETCH_DIST = PREFETCH_ITER * DST_M_BLK;

static inline void _matmul_submat(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n, int dst_line_stride) {
    assert(m % DST_M_BLK == 0);
    assert(n % DST_N_BLK == 0);
    for (int m_idx = 0; m_idx < m; m_idx += DST_M_BLK) {
        for (int n_idx = 0; n_idx < n; n_idx += DST_N_BLK) {
#ifdef __AVX512F__
            const double *lhs_val_ptr = lhs + m_idx * k;
            const double *rhs_vec_ptr = rhs + n_idx * k;
            double *dst_ptr = dst + m_idx * dst_line_stride + n_idx;
            __m512d out00, out10, out20, out30, out40, out50, out60, out70;
            __m512d out01, out11, out21, out31, out41, out51, out61, out71;

#define load_out(i, j) out##i##j = _mm512_loadu_pd(dst_ptr + i * dst_line_stride + j * 8)
            load_out(0, 0), load_out(1, 0), load_out(2, 0), load_out(3, 0);
            load_out(4, 0), load_out(5, 0), load_out(6, 0), load_out(7, 0);
            load_out(0, 1), load_out(1, 1), load_out(2, 1), load_out(3, 1);
            load_out(4, 1), load_out(5, 1), load_out(6, 1), load_out(7, 1);
#undef load_out
            for (int k_idx = 0; k_idx < k; k_idx++) {
#define fma(i, j) out##i##j = _mm512_fmadd_pd(lhs_vbc##i, rhs_vec##j, out##i##j)
                __m512d rhs_vec0 = _mm512_loadu_pd(rhs_vec_ptr + 0);
                __m512d rhs_vec1 = _mm512_loadu_pd(rhs_vec_ptr + 8);
                __m512d lhs_vbc0 = _mm512_set1_pd(lhs_val_ptr[0]);
                __m512d lhs_vbc1 = _mm512_set1_pd(lhs_val_ptr[1]);
                __m512d lhs_vbc2 = _mm512_set1_pd(lhs_val_ptr[2]);
                __m512d lhs_vbc3 = _mm512_set1_pd(lhs_val_ptr[3]);
                fma(0, 0), fma(1, 0), fma(2, 0), fma(3, 0);
                fma(0, 1), fma(1, 1), fma(2, 1), fma(3, 1);
                __m512d lhs_vbc4 = _mm512_set1_pd(lhs_val_ptr[4]);
                __m512d lhs_vbc5 = _mm512_set1_pd(lhs_val_ptr[5]);
                __m512d lhs_vbc6 = _mm512_set1_pd(lhs_val_ptr[6]);
                __m512d lhs_vbc7 = _mm512_set1_pd(lhs_val_ptr[7]);
                fma(4, 0), fma(5, 0), fma(6, 0), fma(7, 0);
                fma(4, 1), fma(5, 1), fma(6, 1), fma(7, 1);
                _mm_prefetch(rhs_vec_ptr + RHS_PREFETCH_DIST + 0, _MM_HINT_T0);
                _mm_prefetch(rhs_vec_ptr + RHS_PREFETCH_DIST + 8, _MM_HINT_T0);
                _mm_prefetch(lhs_val_ptr + LHS_PREFETCH_DIST + 0, _MM_HINT_T0);
                lhs_val_ptr += DST_M_BLK;
                rhs_vec_ptr += DST_N_BLK;
#undef fma
            }
#define store_out(i, j) _mm512_storeu_pd(dst_ptr + i * dst_line_stride + j * 8, out##i##j)
            store_out(0, 0), store_out(1, 0), store_out(2, 0), store_out(3, 0);
            store_out(4, 0), store_out(5, 0), store_out(6, 0), store_out(7, 0);
            store_out(0, 1), store_out(1, 1), store_out(2, 1), store_out(3, 1);
            store_out(4, 1), store_out(5, 1), store_out(6, 1), store_out(7, 1);
#undef store_out
#else
            f64 out[DST_M_BLK * DST_N_BLK];
            for (int i = 0; i < DST_M_BLK; i++) {
                for (int j = 0; j < DST_N_BLK; j++) {
                    out[i * DST_N_BLK + j] = dst[(m_idx + i) * dst_line_stride + (n_idx + j)];
                }
            }
            for (int k_idx = 0; k_idx < k; k_idx++) {
                for (int i = 0; i < DST_M_BLK; i++) {
                    for (int j = 0; j < DST_N_BLK; j++) {
                        out[i * DST_N_BLK + j] +=
                            lhs[m_idx * k + k_idx * DST_M_BLK + i] * rhs[n_idx * k + k_idx * DST_N_BLK + j];
                    }
                }
            }
            for (int i = 0; i < DST_M_BLK; i++) {
                for (int j = 0; j < DST_N_BLK; j++) {
                    dst[(m_idx + i) * dst_line_stride + (n_idx + j)] = out[i * DST_N_BLK + j];
                }
            }
#endif
        }
    }
}

static inline void _matmul_block(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n) {
    f64 *lhs_packed = (f64 *)malloc_aligned((m * k + LHS_PREFETCH_DIST) * sizeof(f64), 64);
    f64 *rhs_packed = (f64 *)malloc_aligned((k * n + RHS_PREFETCH_DIST) * sizeof(f64), 64);

    pack_matrix_lhs(lhs_packed, lhs, m, k, M_BLK, K_BLK);
    pack_matrix_rhs(rhs_packed, rhs, k, n, K_BLK, N_BLK);

    memset(dst, 0, m * n * sizeof(f64));

    for (int m_idx = 0, m_block; m_idx < m; m_idx += m_block) {
        m_block = min_int(M_BLK, m - m_idx);
        for (int n_idx = 0, n_block; n_idx < n; n_idx += n_block) {
            n_block = min_int(N_BLK, n - n_idx);
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

void MatMul4000x16000x128::compute(f64 *dst, const f64 *lhs, const f64 *rhs, int m, int k, int n) const {
    assert(m == M && k == K && n == N);
    _matmul_block(dst, lhs, rhs, m, k, n);
}

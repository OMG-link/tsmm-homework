#include "core/kernels/144x144x144.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "core/matmul.h"
#include "core/pack.h"

static const int OPK_M_BLK = 8;
static const int OPK_N_BLK = 24;

static const int M = 144;
static const int K = 144;
static const int N = 144;

static const int M_BLK = 144;
static const int K_BLK = 144;
static const int N_BLK = 144;

#ifdef __AVX512F__
#define load_out(i, j) out##i##j = _mm512_loadu_pd(dst_ptr + i * dst_line_stride + j * 8)
#define zero_out(i, j) out##i##j = _mm512_setzero_pd()
#define fma(i, j) out##i##j = _mm512_fmadd_pd(lhs_vbc##i, rhs_vec##j, out##i##j)
#define store_out(i, j) _mm512_storeu_pd(dst_ptr + i * dst_line_stride + j * 8, out##i##j)
#endif

static inline void _matmul_submat(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k,
                                  int n, int lhs_line_stride, int rhs_line_strde, int dst_line_stride) {
    assert(m % OPK_M_BLK == 0);
    assert(n % OPK_N_BLK == 0);
    for (int m_idx = 0; m_idx < m; m_idx += OPK_M_BLK) {
        for (int n_idx = 0; n_idx < n; n_idx += OPK_N_BLK) {
#ifdef __AVX512F__
            const f64 *lhs_val0_ptr = lhs + m_idx * lhs_line_stride;
            const f64 *rhs_vec_ptr = rhs + n_idx;
            f64 *dst_ptr = dst + m_idx * dst_line_stride + n_idx;
            __m512d out00, out10, out20, out30, out40, out50, out60, out70;
            __m512d out01, out11, out21, out31, out41, out51, out61, out71;
            __m512d out02, out12, out22, out32, out42, out52, out62, out72;
            zero_out(0, 0), zero_out(1, 0), zero_out(2, 0), zero_out(3, 0);
            zero_out(4, 0), zero_out(5, 0), zero_out(6, 0), zero_out(7, 0);
            zero_out(0, 1), zero_out(1, 1), zero_out(2, 1), zero_out(3, 1);
            zero_out(4, 1), zero_out(5, 1), zero_out(6, 1), zero_out(7, 1);
            zero_out(0, 2), zero_out(1, 2), zero_out(2, 2), zero_out(3, 2);
            zero_out(4, 2), zero_out(5, 2), zero_out(6, 2), zero_out(7, 2);
            for (int k_idx = 0; k_idx < k; k_idx++) {
                const int PREFETCH_ITER = 8;
                const int LHS_PREFETCH_DIST = PREFETCH_ITER * 1;
                const int RHS_PREFETCH_DIST = PREFETCH_ITER * rhs_line_strde;
                __m512d rhs_vec0 = _mm512_loadu_pd(rhs_vec_ptr + 0);
                __m512d rhs_vec1 = _mm512_loadu_pd(rhs_vec_ptr + 8);
                __m512d rhs_vec2 = _mm512_loadu_pd(rhs_vec_ptr + 16);
                __m512d lhs_vbc0 = _mm512_set1_pd(*(lhs_val0_ptr + 0 * lhs_line_stride));
                __m512d lhs_vbc1 = _mm512_set1_pd(*(lhs_val0_ptr + 1 * lhs_line_stride));
                __m512d lhs_vbc2 = _mm512_set1_pd(*(lhs_val0_ptr + 2 * lhs_line_stride));
                __m512d lhs_vbc3 = _mm512_set1_pd(*(lhs_val0_ptr + 3 * lhs_line_stride));
                fma(0, 0), fma(1, 0), fma(2, 0), fma(3, 0);
                fma(0, 1), fma(1, 1), fma(2, 1), fma(3, 1);
                fma(0, 2), fma(1, 2), fma(2, 2), fma(3, 2);
                __m512d lhs_vbc4 = _mm512_set1_pd(*(lhs_val0_ptr + 4 * lhs_line_stride));
                __m512d lhs_vbc5 = _mm512_set1_pd(*(lhs_val0_ptr + 5 * lhs_line_stride));
                __m512d lhs_vbc6 = _mm512_set1_pd(*(lhs_val0_ptr + 6 * lhs_line_stride));
                __m512d lhs_vbc7 = _mm512_set1_pd(*(lhs_val0_ptr + 7 * lhs_line_stride));
                fma(4, 0), fma(5, 0), fma(6, 0), fma(7, 0);
                fma(4, 1), fma(5, 1), fma(6, 1), fma(7, 1);
                fma(4, 2), fma(5, 2), fma(6, 2), fma(7, 2);
                _mm_prefetch(rhs_vec_ptr + RHS_PREFETCH_DIST + 0, _MM_HINT_T0);
                _mm_prefetch(rhs_vec_ptr + RHS_PREFETCH_DIST + 8, _MM_HINT_T0);
                _mm_prefetch(rhs_vec_ptr + RHS_PREFETCH_DIST + 16, _MM_HINT_T0);
                if (k_idx % 8 == 0) {
                    _mm_prefetch(lhs_val0_ptr + 0 * lhs_line_stride + LHS_PREFETCH_DIST, _MM_HINT_T0);
                    _mm_prefetch(lhs_val0_ptr + 1 * lhs_line_stride + LHS_PREFETCH_DIST, _MM_HINT_T0);
                    _mm_prefetch(lhs_val0_ptr + 2 * lhs_line_stride + LHS_PREFETCH_DIST, _MM_HINT_T0);
                    _mm_prefetch(lhs_val0_ptr + 3 * lhs_line_stride + LHS_PREFETCH_DIST, _MM_HINT_T0);
                    _mm_prefetch(lhs_val0_ptr + 4 * lhs_line_stride + LHS_PREFETCH_DIST, _MM_HINT_T0);
                    _mm_prefetch(lhs_val0_ptr + 5 * lhs_line_stride + LHS_PREFETCH_DIST, _MM_HINT_T0);
                    _mm_prefetch(lhs_val0_ptr + 6 * lhs_line_stride + LHS_PREFETCH_DIST, _MM_HINT_T0);
                    _mm_prefetch(lhs_val0_ptr + 7 * lhs_line_stride + LHS_PREFETCH_DIST, _MM_HINT_T0);
                }
                lhs_val0_ptr += 1;
                rhs_vec_ptr += rhs_line_strde;
            }
            store_out(0, 0), store_out(1, 0), store_out(2, 0), store_out(3, 0);
            store_out(4, 0), store_out(5, 0), store_out(6, 0), store_out(7, 0);
            store_out(0, 1), store_out(1, 1), store_out(2, 1), store_out(3, 1);
            store_out(4, 1), store_out(5, 1), store_out(6, 1), store_out(7, 1);
            store_out(0, 2), store_out(1, 2), store_out(2, 2), store_out(3, 2);
            store_out(4, 2), store_out(5, 2), store_out(6, 2), store_out(7, 2);
#else
            f64 out[OPK_M_BLK * OPK_N_BLK] = {0};
            for (int k_idx = 0; k_idx < k; k_idx++) {
                for (int i = 0; i < OPK_M_BLK; i++) {
                    for (int j = 0; j < OPK_N_BLK; j++) {
                        out[i * OPK_N_BLK + j] +=
                            lhs[(m_idx + i) * lhs_line_stride + k_idx] * rhs[k_idx * rhs_line_strde + (n_idx + j)];
                    }
                }
            }
            for (int i = 0; i < OPK_M_BLK; i++) {
                for (int j = 0; j < OPK_N_BLK; j++) {
                    dst[(m_idx + i) * dst_line_stride + (n_idx + j)] = out[i * OPK_N_BLK + j];
                }
            }
#endif
        }
    }
}

static inline void _matmul_block(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k,
                                 int n) {
    _matmul_submat(dst, lhs, rhs, m, k, n, k, n, n);
}

void MatMul144x144x144::compute(f64 *RESTRICT dst, const f64 *RESTRICT lhs, const f64 *RESTRICT rhs, int m, int k,
                                int n) const {
    assert(m == M && k == K && n == N);
    _matmul_block(dst, lhs, rhs, M, K, N);
}

#ifdef __AVX512F__
#undef load_out
#undef zero_out
#undef fma
#undef store_out
#endif
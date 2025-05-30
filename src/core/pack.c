#include "core/pack.h"

#include <assert.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

static inline void reorder_matrix_lhs(f64 *RESTRICT dst, const f64 *RESTRICT src, int m, int k, int src_line_stride,
                                      const int OPK_M_BLK) {
    const int M_BLK = OPK_M_BLK;
    f64 *dst_ptr = dst;

#ifdef __AVX512F__
    __m512i offsets =
        _mm512_set_epi64(7 * src_line_stride, 6 * src_line_stride, 5 * src_line_stride, 4 * src_line_stride,
                         3 * src_line_stride, 2 * src_line_stride, 1 * src_line_stride, 0 * src_line_stride);
    int m_idx;
    for (m_idx = 0; m_idx + M_BLK <= m; m_idx += M_BLK) {
        assert(M_BLK == 8);
        const int K_BLK = 8;
        int k_idx = 0;
        const f64 *src_l0 = src + (m_idx + 0) * src_line_stride;
        const f64 *src_l1 = src + (m_idx + 1) * src_line_stride;
        const f64 *src_l2 = src + (m_idx + 2) * src_line_stride;
        const f64 *src_l3 = src + (m_idx + 3) * src_line_stride;
        const f64 *src_l4 = src + (m_idx + 4) * src_line_stride;
        const f64 *src_l5 = src + (m_idx + 5) * src_line_stride;
        const f64 *src_l6 = src + (m_idx + 6) * src_line_stride;
        const f64 *src_l7 = src + (m_idx + 7) * src_line_stride;
        for (; k_idx + K_BLK <= k; k_idx += K_BLK) {
            // 0. Prefetch
            const int PREFETCH_DIST = 8 * K_BLK;
            if (k_idx + PREFETCH_DIST + K_BLK <= k) {
                _mm_prefetch(src_l0 + PREFETCH_DIST, _MM_HINT_T0);
                _mm_prefetch(src_l1 + PREFETCH_DIST, _MM_HINT_T0);
                _mm_prefetch(src_l2 + PREFETCH_DIST, _MM_HINT_T0);
                _mm_prefetch(src_l3 + PREFETCH_DIST, _MM_HINT_T0);
                _mm_prefetch(src_l4 + PREFETCH_DIST, _MM_HINT_T0);
                _mm_prefetch(src_l5 + PREFETCH_DIST, _MM_HINT_T0);
                _mm_prefetch(src_l6 + PREFETCH_DIST, _MM_HINT_T0);
                _mm_prefetch(src_l7 + PREFETCH_DIST, _MM_HINT_T0);
            }

            // 1. Load 8 rows × 8 columns
            __m512d row0 = _mm512_loadu_pd(src_l0);
            __m512d row1 = _mm512_loadu_pd(src_l1);
            __m512d row2 = _mm512_loadu_pd(src_l2);
            __m512d row3 = _mm512_loadu_pd(src_l3);
            __m512d row4 = _mm512_loadu_pd(src_l4);
            __m512d row5 = _mm512_loadu_pd(src_l5);
            __m512d row6 = _mm512_loadu_pd(src_l6);
            __m512d row7 = _mm512_loadu_pd(src_l7);

            // 2. Transpose 8x8 block in registers
            __m512d t0 = _mm512_unpacklo_pd(row0, row1);
            __m512d t1 = _mm512_unpackhi_pd(row0, row1);
            __m512d t2 = _mm512_unpacklo_pd(row2, row3);
            __m512d t3 = _mm512_unpackhi_pd(row2, row3);
            __m512d t4 = _mm512_unpacklo_pd(row4, row5);
            __m512d t5 = _mm512_unpackhi_pd(row4, row5);
            __m512d t6 = _mm512_unpacklo_pd(row6, row7);
            __m512d t7 = _mm512_unpackhi_pd(row6, row7);

            __m512d s0 = _mm512_shuffle_f64x2(t0, t2, 0x88); // low128(t0), low128(t2)
            __m512d s1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
            __m512d s2 = _mm512_shuffle_f64x2(t0, t2, 0xdd); // high128(t0), high128(t2)
            __m512d s3 = _mm512_shuffle_f64x2(t1, t3, 0xdd);
            __m512d s4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
            __m512d s5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
            __m512d s6 = _mm512_shuffle_f64x2(t4, t6, 0xdd);
            __m512d s7 = _mm512_shuffle_f64x2(t5, t7, 0xdd);

            __m512d q0 = _mm512_shuffle_f64x2(s0, s4, 0x88); // low256(s0), low256(s4)
            __m512d q1 = _mm512_shuffle_f64x2(s1, s5, 0x88);
            __m512d q2 = _mm512_shuffle_f64x2(s2, s6, 0x88);
            __m512d q3 = _mm512_shuffle_f64x2(s3, s7, 0x88);
            __m512d q4 = _mm512_shuffle_f64x2(s0, s4, 0xdd); // high256(s0), high256(s4)
            __m512d q5 = _mm512_shuffle_f64x2(s1, s5, 0xdd);
            __m512d q6 = _mm512_shuffle_f64x2(s2, s6, 0xdd);
            __m512d q7 = _mm512_shuffle_f64x2(s3, s7, 0xdd);

            // 3. Store columns (already column-major packed)
            _mm512_storeu_pd(dst_ptr + 0 * 8, q0);
            _mm512_storeu_pd(dst_ptr + 1 * 8, q1);
            _mm512_storeu_pd(dst_ptr + 2 * 8, q2);
            _mm512_storeu_pd(dst_ptr + 3 * 8, q3);
            _mm512_storeu_pd(dst_ptr + 4 * 8, q4);
            _mm512_storeu_pd(dst_ptr + 5 * 8, q5);
            _mm512_storeu_pd(dst_ptr + 6 * 8, q6);
            _mm512_storeu_pd(dst_ptr + 7 * 8, q7);

            src_l0 += K_BLK;
            src_l1 += K_BLK;
            src_l2 += K_BLK;
            src_l3 += K_BLK;
            src_l4 += K_BLK;
            src_l5 += K_BLK;
            src_l6 += K_BLK;
            src_l7 += K_BLK;

            dst_ptr += 64; // 8*8 f64 elements
        }
        for (; k_idx < k; k_idx++) {
            const f64 *src_ptr = src + m_idx * src_line_stride + k_idx;
            __m512d v = _mm512_i64gather_pd(offsets, src_ptr, sizeof(f64));
            _mm512_storeu_pd(dst_ptr, v);
            dst_ptr += 8;
        }
    }
    for (int k_idx = 0; k_idx < k; k_idx++) {
        for (int m_idx2 = m_idx; m_idx2 < m; m_idx2++) {
            *dst_ptr++ = *(src + m_idx2 * src_line_stride + k_idx);
        }
    }
#else
    for (int m_idx = 0, m_block; m_idx < m; m_idx += m_block) {
        m_block = min_int(M_BLK, m - m_idx);
        for (int k_idx = 0; k_idx < k; k_idx++) {
            for (int m_idx2 = 0; m_idx2 < m_block; m_idx2++) {
                *dst_ptr++ = *(src + (m_idx + m_idx2) * src_line_stride + k_idx);
            }
        }
    }
#endif
}

void pack_matrix_lhs(f64 *RESTRICT dst, const f64 *RESTRICT src, int m, int k, const int M_BLK, const int K_BLK,
                     const int OPK_M_BLK) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int m_idx = 0; m_idx < m; m_idx += M_BLK) {
        for (int k_idx = 0; k_idx < k; k_idx += K_BLK) {
            int m_block = min_int(M_BLK, m - m_idx);
            int k_block = min_int(K_BLK, k - k_idx);
            const f64 *src_ptr = src + m_idx * k + k_idx;
            f64 *dst_ptr = dst + m_idx * k + k_idx * m_block;
            reorder_matrix_lhs(dst_ptr, src_ptr, m_block, k_block, k, OPK_M_BLK);
        }
    }
}

static inline void reorder_matrix_rhs(f64 *RESTRICT dst, const f64 *RESTRICT src, int k, int n, int src_line_stride,
                                      const int OPK_N_BLK) {
    const int N_BLK = OPK_N_BLK;
    f64 *dst_ptr = dst;
#ifdef __AVX512F__
    int n_idx;
    for (n_idx = 0; n_idx + N_BLK <= n; n_idx += N_BLK) {
        for (int k_idx = 0; k_idx < k; k_idx++) {
            for (int n_idx2 = 0; n_idx2 < N_BLK; n_idx2 += 8) {
                const f64 *src_ptr = src + k_idx * src_line_stride + n_idx + n_idx2;
                __m512d v = _mm512_loadu_pd(src_ptr);
                _mm512_storeu_pd(dst_ptr, v);
                dst_ptr += 8;
            }
        }
    }
    for (int k_idx = 0; k_idx < k; k_idx++) {
        for (int n_idx2 = n_idx; n_idx2 < n; n_idx2++) {
            *dst_ptr++ = *(src + k_idx * src_line_stride + n_idx2);
        }
    }
#else
    for (int n_idx = 0, n_block; n_idx < n; n_idx += n_block) {
        n_block = min_int(N_BLK, n - n_idx);
        for (int k_idx = 0; k_idx < k; k_idx++) {
            for (int n_idx2 = 0; n_idx2 < n_block; n_idx2++) {
                *dst_ptr++ = *(src + k_idx * src_line_stride + (n_idx + n_idx2));
            }
        }
    }
#endif
}

void pack_matrix_rhs(f64 *RESTRICT dst, const f64 *RESTRICT src, int k, int n, const int K_BLK, const int N_BLK,
                     const int OPK_N_BLK) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int k_idx = 0; k_idx < k; k_idx += K_BLK) {
        for (int n_idx = 0; n_idx < n; n_idx += N_BLK) {
            int k_block = min_int(K_BLK, k - k_idx);
            int n_block = min_int(N_BLK, n - n_idx);
            const f64 *src_ptr = src + k_idx * n + n_idx;
            f64 *dst_ptr = dst + k_idx * n_block + n_idx * k;
            reorder_matrix_rhs(dst_ptr, src_ptr, k_block, n_block, n, OPK_N_BLK);
        }
    }
}
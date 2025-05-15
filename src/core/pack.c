#include "core/pack.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

static inline int min(int a, int b) { return a < b ? a : b; }

static void reorder_matrix_lhs(f64 *dst, const f64 *src, int m, int k, int src_line_stride) {
    const int M_BLK = DST_M_BLK;
    f64 *dst_ptr = dst;

#ifdef __AVX512F__
    __m512i offsets = _mm512_set_epi64( //
        7 * src_line_stride,            //
        6 * src_line_stride,            //
        5 * src_line_stride,            //
        4 * src_line_stride,            //
        3 * src_line_stride,            //
        2 * src_line_stride,            //
        1 * src_line_stride,            //
        0 * src_line_stride             //
    );
    int m_idx;
    for (m_idx = 0; m_idx + M_BLK <= m; m_idx += M_BLK) {
        for (int k_idx = 0; k_idx < k; k_idx++) {
            for (int m_idx2 = 0; m_idx2 < M_BLK; m_idx2 += 8) {
                const f64 *src_ptr = src + (m_idx + m_idx2) * src_line_stride + k_idx;
                __m512d v = _mm512_i64gather_pd(offsets, src_ptr, sizeof(f64));
                _mm512_storeu_pd(dst_ptr, v);
                dst_ptr += 8;
            }
        }
    }
    for (int k_idx = 0; k_idx < k; k_idx++) {
        for (int m_idx2 = m_idx; m_idx2 < m; m_idx2++) {
            *dst_ptr++ = *(src + m_idx2 * src_line_stride + k_idx);
        }
    }
#else
    for (int m_idx = 0, m_block; m_idx < m; m_idx += m_block) {
        m_block = min(M_BLK, m - m_idx);
        for (int k_idx = 0; k_idx < k; k_idx++) {
            for (int m_idx2 = 0; m_idx2 < m_block; m_idx2++) {
                *dst_ptr++ = *(src + (m_idx + m_idx2) * src_line_stride + k_idx);
            }
        }
    }
#endif
}

void pack_matrix_lhs(f64 *dst, const f64 *src, int m, int k, const int M_BLK, const int K_BLK) {
    for (int m_idx = 0, m_block; m_idx < m; m_idx += m_block) {
        m_block = min(M_BLK, m - m_idx);
        for (int k_idx = 0, k_block; k_idx < k; k_idx += k_block) {
            k_block = min(K_BLK, k - k_idx);
            const f64 *src_ptr = src + m_idx * k + k_idx;
            f64 *dst_ptr = dst + m_idx * k + k_idx * m_block;
            reorder_matrix_lhs(dst_ptr, src_ptr, m_block, k_block, k);
        }
    }
}

static void reorder_matrix_rhs(f64 *dst, const f64 *src, int k, int n, int src_line_stride) {
    const int N_BLK = DST_N_BLK;
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
        n_block = min(N_BLK, n - n_idx);
        for (int k_idx = 0; k_idx < k; k_idx++) {
            for (int n_idx2 = 0; n_idx2 < n_block; n_idx2++) {
                *dst_ptr++ = *(src + k_idx * src_line_stride + (n_idx + n_idx2));
            }
        }
    }
#endif
}

void pack_matrix_rhs(f64 *dst, const f64 *src, int k, int n, const int K_BLK, const int N_BLK) {
    for (int k_idx = 0, k_block; k_idx < k; k_idx += k_block) {
        k_block = min(K_BLK, k - k_idx);
        for (int n_idx = 0, n_block; n_idx < n; n_idx += n_block) {
            n_block = min(N_BLK, n - n_idx);
            const f64 *src_ptr = src + k_idx * n + n_idx;
            f64 *dst_ptr = dst + k_idx * n_block + n_idx * k;
            reorder_matrix_rhs(dst_ptr, src_ptr, k_block, n_block, n);
        }
    }
}
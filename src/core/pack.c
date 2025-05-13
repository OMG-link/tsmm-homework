#include "core/pack.h"

static inline int min(int a, int b) { return a < b ? a : b; }

static void reorder_matrix_lhs(f64 *dst, const f64 *src, int m, int k, int src_line_stride) {
    const int M_BLK = DST_M_BLK;
    f64 *dst_ptr = dst;
    for (int m_idx = 0, m_block; m_idx < m; m_idx += m_block) {
        m_block = min(M_BLK, m - m_idx);
        for (int k_idx = 0; k_idx < k; k_idx++) {
            // todo: use strided load and vectorized store
            for (int m_idx2 = 0; m_idx2 < m_block; m_idx2++) {
                *dst_ptr++ = *(src + (m_idx + m_idx2) * src_line_stride + k_idx);
            }
        }
    }
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
    for (int n_idx = 0, n_block; n_idx < n; n_idx += n_block) {
        n_block = min(N_BLK, n - n_idx);
        for (int k_idx = 0; k_idx < k; k_idx++) {
            // todo: use vectorized load and store
            for (int n_idx2 = 0; n_idx2 < n_block; n_idx2++) {
                *dst_ptr++ = *(src + k_idx * src_line_stride + (n_idx + n_idx2));
            }
        }
    }
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
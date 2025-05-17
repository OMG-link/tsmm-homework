#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/matmul.h"
#include "matrix.hpp"

void test(int m, int k, int n) {
    using namespace std::chrono;

    f64 *dst = (f64 *)malloc_aligned(m * n * sizeof(f64), 64);
    f64 *lhs = (f64 *)malloc_aligned(m * k * sizeof(f64), 64);
    f64 *rhs = (f64 *)malloc_aligned(k * n * sizeof(f64), 64);

    const int M_BLK = 128;
    const int K_BLK = 256;
    const int N_BLK = 128;

    auto start_submatmul = high_resolution_clock::now();
    for (int m_idx = 0, m_block; m_idx < m; m_idx += m_block) {
        m_block = min_int(M_BLK, m - m_idx);
        for (int n_idx = 0, n_block; n_idx < n; n_idx += n_block) {
            n_block = min_int(N_BLK, n - n_idx);
            for (int k_idx = 0, k_block; k_idx < k; k_idx += k_block) {
                k_block = min_int(K_BLK, k - k_idx);
                const f64 *lhs_submat = lhs + m_idx * k + k_idx * m_block;
                const f64 *rhs_submat = rhs + n_idx * k + k_idx * n_block;
                f64 *dst_submat = dst + m_idx * n + n_idx;
                matmul_submat(dst_submat, lhs_submat, rhs_submat, m_block, k_block, n_block, n);
            }
        }
    }
    auto end_submatmul = high_resolution_clock::now();
    auto duration_submatmul = duration_cast<duration<double>>(end_submatmul - start_submatmul);

    double seconds = duration_submatmul.count();
    double gflops = (2.0 * m * k * n) / (seconds * 1e9);
    printf("ok | submatmul = %.6f s | GFLOPS = %.2f\n", seconds, gflops);

    free(lhs);
    free(rhs);
    free(dst);
}

int main() { test(4000, 16000, 128); }
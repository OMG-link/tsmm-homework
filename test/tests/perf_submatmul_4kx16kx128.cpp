#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/matmul.h"
#include "matrix.hpp"

static inline int min(int a, int b) { return a < b ? a : b; }

void test(int m, int k, int n) {
    using namespace std::chrono;

    f64 *lhs, *rhs, *dst;

    if (posix_memalign((void **)&dst, 64, m * n * sizeof(f64)) != 0) {
        perror("posix_memalign failed for dst");
        exit(1);
    }
    if (posix_memalign((void **)&lhs, 64, m * k * sizeof(f64)) != 0) {
        perror("posix_memalign failed for lhs");
        exit(1);
    }
    if (posix_memalign((void **)&rhs, 64, k * n * sizeof(f64)) != 0) {
        perror("posix_memalign failed for rhs");
        exit(1);
    }

    auto start_submatmul = high_resolution_clock::now();
    int M_BLK = 128;
    int K_BLK = 256;
    int N_BLK = 128;
    for (int m_idx = 0, m_block; m_idx < m; m_idx += m_block) {
        m_block = min(M_BLK, m - m_idx);
        for (int n_idx = 0, n_block; n_idx < n; n_idx += n_block) {
            n_block = min(N_BLK, n - n_idx);
            for (int k_idx = 0, k_block; k_idx < k; k_idx += k_block) {
                k_block = min(K_BLK, k - k_idx);
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
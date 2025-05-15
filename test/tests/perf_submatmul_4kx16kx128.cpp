#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/matmul.h"
#include "matrix.hpp"

static inline int min(int a, int b) { return a < b ? a : b; }

void test(const Matrix &a, const Matrix &b) {
    using namespace std::chrono;

    f64 *dst = (f64 *)malloc(a.rows() * b.cols() * sizeof(f64));

    auto start_submatmul = high_resolution_clock::now();
    int m = a.rows();
    int k = a.cols();
    int n = b.cols();
    int M_BLK = 256;
    int K_BLK = 256;
    int N_BLK = 128;
    for (int m_idx = 0, m_block; m_idx < m; m_idx += m_block) {
        m_block = min(M_BLK, m - m_idx);
        for (int n_idx = 0, n_block; n_idx < n; n_idx += n_block) {
            n_block = min(N_BLK, n - n_idx);
            for (int k_idx = 0, k_block; k_idx < k; k_idx += k_block) {
                k_block = min(K_BLK, k - k_idx);
                const f64 *lhs_submat = a.data() + m_idx * k + k_idx * m_block;
                const f64 *rhs_submat = b.data() + n_idx * k + k_idx * n_block;
                f64 *dst_submat = dst + m_idx * n + n_idx;
                matmul_submat(dst_submat, lhs_submat, rhs_submat, m_block, k_block, n_block, n);
            }
        }
    }
    auto end_submatmul = high_resolution_clock::now();
    auto duration_submatmul = duration_cast<milliseconds>(end_submatmul - start_submatmul).count();

    printf("ok | submatmul = %ldms\n", duration_submatmul);

    free(dst);
}

int main() {
    Matrix a(4000, 16000), b(16000, 128);
    test(a, b);
}
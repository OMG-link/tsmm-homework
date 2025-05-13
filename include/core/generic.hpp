#pragma once
#include "matrix.hpp"

Matrix matmul_generic(const Matrix &A, const Matrix &B) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(B.cols());
    const int k = static_cast<int>(A.cols());
    Matrix C(m, n);
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            f64 sum = 0;
            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                sum += A.at(m_idx, k_idx) * B.at(k_idx, n_idx);
            }
            C.at(m_idx, n_idx) = sum;
        }
    }
    return C;
}
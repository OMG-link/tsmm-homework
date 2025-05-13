#include "matmul.hpp"
#include "dispatcher.hpp"

Matrix matmul(const Matrix &lhs, const Matrix &rhs) {
    static MatMulDispatcher dispatcher;
    assert(lhs.cols() == rhs.rows());
    return dispatcher(lhs, rhs);
}

Matrix matmul_generic(const Matrix &lhs, const Matrix &rhs) {
    const size_t m = lhs.rows();
    const size_t n = rhs.cols();
    const size_t k = lhs.cols();
    Matrix dst(m, n);
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            f64 sum = 0;
            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                sum += lhs.at(m_idx, k_idx) * rhs.at(k_idx, n_idx);
            }
            dst.at(m_idx, n_idx) = sum;
        }
    }
    return dst;
}
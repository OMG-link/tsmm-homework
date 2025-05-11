#pragma once
#include "matrix.hpp"

Matrix matmul_generic(const Matrix &A, const Matrix &B) {
    const int M = static_cast<int>(A.rows());
    const int N = static_cast<int>(B.cols());
    const int K = static_cast<int>(A.cols());
    Matrix C(M, N);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            f64 sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
    return C;
}
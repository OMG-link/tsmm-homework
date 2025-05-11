#pragma once
#include <memory>
#include <vector>

#include "core/generic.hpp"
#include "core/kernels/4000x16000x128.hpp"
#include "core/optimized.hpp"
#include "matrix.hpp"

class MatMulDispatcher {
  public:
    MatMulDispatcher() { register_kernel<MatMul4000x16000x128>(); }

    Matrix operator()(const Matrix &A, const Matrix &B) {
        const int M = static_cast<int>(A.rows());
        const int N = static_cast<int>(B.cols());
        const int K = static_cast<int>(A.cols());

        Matrix C(M, N);

        for (const auto &kernel : kernels_) {
            if (kernel->match(M, N, K)) {
                kernel->compute(A.data(), B.data(), C.data(), M, N, K);
                return C;
            }
        }

        return matmul_generic(A, B);
    }

  private:
    template <typename Kernel> void register_kernel() { kernels_.emplace_back(std::make_unique<Kernel>()); }

    std::vector<std::unique_ptr<MatMulOptimizedKernel>> kernels_;
};

Matrix matmul(const Matrix &A, const Matrix &B) {
    static MatMulDispatcher dispatcher;
    assert(A.cols() == B.rows());
    return dispatcher(A, B);
}
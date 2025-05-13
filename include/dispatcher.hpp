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

    Matrix operator()(const Matrix &lhs, const Matrix &rhs) {
        const int m = static_cast<int>(lhs.rows());
        const int k = static_cast<int>(lhs.cols());
        const int n = static_cast<int>(rhs.cols());

        Matrix lhs_rm = lhs;
        Matrix rhs_cm = rhs;
        lhs_rm.set_store_mode(ROW_MAJOR);
        rhs_cm.set_store_mode(COL_MAJOR);

        Matrix dst(m, n);

        for (const auto &kernel : kernels_) {
            if (kernel->match(m, k, n)) {
                kernel->compute(lhs_rm.data(), rhs_cm.data(), dst.data(), m, k, n);
                return dst;
            }
        }

        return matmul_generic(lhs, rhs);
    }

  private:
    template <typename Kernel> void register_kernel() { kernels_.emplace_back(std::make_unique<Kernel>()); }

    std::vector<std::unique_ptr<MatMulOptimizedKernel>> kernels_;
};

Matrix matmul(const Matrix &lhs, const Matrix &rhs) {
    static MatMulDispatcher dispatcher;
    assert(lhs.cols() == rhs.rows());
    return dispatcher(lhs, rhs);
}
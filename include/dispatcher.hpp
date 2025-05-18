#pragma once
#include <memory>
#include <vector>

#include "core/kernel_base.hpp"
#include "core/kernels/4000x16000x128.hpp"
#include "core/kernels/8x16x16000.hpp"
#include "matmul.hpp"
#include "matrix.hpp"

class MatMulDispatcher {
  public:
    MatMulDispatcher() {
        register_kernel<MatMul4000x16000x128>();
        register_kernel<MatMul8x16x16000>();
    }

    Matrix operator()(const Matrix &lhs, const Matrix &rhs) {
        const int m = static_cast<int>(lhs.rows());
        const int k = static_cast<int>(lhs.cols());
        const int n = static_cast<int>(rhs.cols());

        const Matrix *lhs_used = &lhs;
        const Matrix *rhs_used = &rhs;

        // copy matrix only when necessary
        Matrix lhs_rm(0, 0), rhs_rm(0, 0);
        if (lhs.store_mode() != ROW_MAJOR) {
            lhs_rm = lhs;
            lhs_rm.set_store_mode(ROW_MAJOR);
            lhs_used = &lhs_rm;
        }
        if (rhs.store_mode() != ROW_MAJOR) {
            rhs_rm = rhs;
            rhs_rm.set_store_mode(ROW_MAJOR);
            rhs_used = &rhs_rm;
        }

        Matrix dst(m, n, ROW_MAJOR);

        for (const auto &kernel : kernels_) {
            if (kernel->match(m, k, n)) {
                kernel->compute(dst.data(), lhs_used->data(), rhs_used->data(), m, k, n);
                return dst;
            }
        }

        return matmul_generic(lhs, rhs);
    }

  private:
    template <typename Kernel> void register_kernel() { kernels_.emplace_back(std::make_unique<Kernel>()); }

    std::vector<std::unique_ptr<MatmulKernelBase>> kernels_;
};

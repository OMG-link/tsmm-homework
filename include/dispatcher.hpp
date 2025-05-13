#pragma once
#include <memory>
#include <vector>

#include "core/kernel_base.hpp"
#include "core/kernels/4000x16000x128.hpp"
#include "matmul.hpp"
#include "matrix.hpp"

class MatMulDispatcher {
  public:
    MatMulDispatcher() { register_kernel<MatMul4000x16000x128>(); }

    Matrix operator()(const Matrix &lhs, const Matrix &rhs) {
        const int m = static_cast<int>(lhs.rows());
        const int k = static_cast<int>(lhs.cols());
        const int n = static_cast<int>(rhs.cols());

        Matrix lhs_rm = lhs;
        Matrix rhs_rm = rhs;
        lhs_rm.set_store_mode(ROW_MAJOR);
        rhs_rm.set_store_mode(ROW_MAJOR);

        Matrix dst(m, n, ROW_MAJOR);

        for (const auto &kernel : kernels_) {
            if (kernel->match(m, k, n)) {
                kernel->compute(dst.data(), lhs_rm.data(), rhs_rm.data(), m, k, n);
                return dst;
            }
        }

        return matmul_generic(lhs, rhs);
    }

  private:
    template <typename Kernel> void register_kernel() { kernels_.emplace_back(std::make_unique<Kernel>()); }

    std::vector<std::unique_ptr<MatmulKernelBase>> kernels_;
};

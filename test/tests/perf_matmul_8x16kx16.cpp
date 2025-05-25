#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/kernels/8x16000x16.hpp"
#include "core/matmul.h"

#include "perf_matmul.hpp"

void test(const Matrix &a, const Matrix &b) {
    test_kernel(
        a, b,
        [](f64 *dst, const f64 *lhs, const f64 *rhs, size_t m, size_t k, size_t n) {
            MatMul8x16000x16 kernel;
            kernel.compute(dst, lhs, rhs, m, k, n);
        },
        "kernel-8x16000x16");
    test_kernel(
        a, b,
        [](f64 *dst, const f64 *lhs, const f64 *rhs, size_t m, size_t k, size_t n) {
            matmul_block(dst, lhs, rhs, m, k, n, 192, 256, 192);
        },
        "generic-optimized");
}

int main() {
    printf("Running on CPU %d.\n", get_cpu_id());
    Matrix a(8, 16000), b(16000, 16);
    test(a, b);
}
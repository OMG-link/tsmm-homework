#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/kernels/32x16x16000.hpp"
#include "core/matmul.h"

#include "perf_matmul.hpp"

void test(const Matrix &a, const Matrix &b) {
    flush_l3_cache();
    test_kernel(
        a, b,
        [](f64 *dst, const f64 *lhs, const f64 *rhs, size_t m, size_t k, size_t n) {
            matmul_block(dst, lhs, rhs, m, k, n, 192, 256, 192);
        },
        "generic-optimized");
    flush_l3_cache();
    test_kernel(
        a, b,
        [](f64 *dst, const f64 *lhs, const f64 *rhs, size_t m, size_t k, size_t n) {
            MatMul32x16x16000 kernel;
            kernel.compute(dst, lhs, rhs, m, k, n);
        },
        "kernel-32x16x16000");
}

int main() {
    printf("Running on CPU %d.\n", get_cpu_id());
    Matrix a(32, 16), b(16, 16000);
    test(a, b);
}
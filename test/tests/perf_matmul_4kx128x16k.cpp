#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/kernels/4000x128x16000.hpp"
#include "core/matmul.h"

#include "perf_matmul.hpp"

void test(const Matrix &a, const Matrix &b) {
    test_kernel(
        a, b,
        [](f64 *dst, const f64 *lhs, const f64 *rhs, size_t m, size_t k, size_t n) {
            MatMul4000x128x16000 kernel;
            kernel.compute(dst, lhs, rhs, m, k, n);
        },
        "kernel-4000x128x16000");
    test_kernel(
        a, b,
        [](f64 *dst, const f64 *lhs, const f64 *rhs, size_t m, size_t k, size_t n) {
            matmul_block(dst, lhs, rhs, m, k, n, 4000, 128, 720);
        },
        "generic-optimized");
}

int main() {
    printf("Running on CPU %d.\n", get_cpu_id());
    Matrix a(4000, 128), b(128, 16000);
    test(a, b);
}
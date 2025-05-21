#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/pack.h"
#include "matrix.hpp"
#include "perf.h"

void test(const Matrix &a, const Matrix &b) {
    using namespace std::chrono;

    f64 *a_packed = (f64 *)malloc_aligned(a.rows() * a.cols() * sizeof(f64), 128);
    f64 *b_packed = (f64 *)malloc_aligned(b.rows() * b.cols() * sizeof(f64), 128);

    auto start_lhs = high_resolution_clock::now();
    pack_matrix_lhs(a_packed, a.data(), a.rows(), a.cols(), 128, 256, 8);
    auto end_lhs = high_resolution_clock::now();
    auto duration_lhs = duration_cast<milliseconds>(end_lhs - start_lhs).count();

    auto start_rhs = high_resolution_clock::now();
    pack_matrix_rhs(b_packed, b.data(), b.rows(), b.cols(), 256, 128, 24);
    auto end_rhs = high_resolution_clock::now();
    auto duration_rhs = duration_cast<milliseconds>(end_rhs - start_rhs).count();

    printf("ok | lhs = %ldms, rhs = %ldms\n", duration_lhs, duration_rhs);

    free(a_packed);
    free(b_packed);
}

int main() {
    printf("Running on CPU %d.\n", get_cpu_id());
    Matrix a(4000, 16000), b(16000, 128);
    test(a, b);
}
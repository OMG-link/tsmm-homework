#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/kernels/32x16000x16.hpp"
#include "matmul.hpp"

void test(const Matrix &a, const Matrix &b) {
    using namespace std::chrono;

    MatMul32x16000x16 kernel;
    Matrix c(a.rows(), b.cols());

    auto start_opt = high_resolution_clock::now();
    kernel.compute(c.data(), a.data(), b.data(), a.rows(), a.cols(), b.cols());
    auto end_opt = high_resolution_clock::now();
    auto duration_opt = duration_cast<duration<double>>(end_opt - start_opt).count();

    double M = static_cast<double>(a.rows());
    double N = static_cast<double>(b.cols());
    double K = static_cast<double>(a.cols());
    double operations = 2.0 * M * N * K;
    double gflops = operations / (duration_opt * 1e9);

    printf("ok | time = %.6e s | GFLOPS = %.2f\n", duration_opt, gflops);
}

int main() {
    Matrix a(32, 16000), b(16000, 16);
    test(a, b);
}
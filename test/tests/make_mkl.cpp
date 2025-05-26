
#include <iostream>
#include <chrono>
#include <mkl.h>

#include <mkl_service.h>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "用法: " << argv[0] << "  <m> <n> <k>\n"
                  << "       A: m×k   B: k×n   C: m×n\n";
        return 1;
    }
    const int m = std::stoi(argv[1]);   // 行数
    const int n = std::stoi(argv[2]);   // 列数
    const int k = std::stoi(argv[3]);   // 中间维

    std::cout << "A: " << m << "×" << k
              << "  B: " << k << "×" << n
              << "  C: " << m << "×" << n << '\n';

    /* ---------- 64 B 对齐分配 ---------- */
    double* A = (double*)mkl_malloc(static_cast<size_t>(m) * k * sizeof(double), 64);
    double* B = (double*)mkl_malloc(static_cast<size_t>(k) * n * sizeof(double), 64);
    double* C = (double*)mkl_malloc(static_cast<size_t>(m) * n * sizeof(double), 64);
    if (!A || !B || !C) {
        std::cerr << "内存分配失败，可能是矩阵太大\n";
        return 2;
    }

    /* ---------- 初始化 ---------- */
#pragma omp parallel for
    for (long long i = 0; i < 1LL * m * k; ++i) A[i] = 1.0;
#pragma omp parallel for
    for (long long i = 0; i < 1LL * k * n; ++i) B[i] = 1.0;
#pragma omp parallel for
    for (long long i = 0; i < 1LL * m * n; ++i) C[i] = 0.0;
    
    mkl_set_num_threads(<thread_num>);

    /* ---------- 热身一次 ---------- */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, A, k, B, n, 0.0, C, n);

    /* ---------- 计时 ---------- */
    auto t0 = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gflops = 2.0 * m * n * k / (elapsed_ms * 1e6);

    std::cout << "C[0] = " << C[0] << '\n'
              << "耗时: " << elapsed_ms << " ms   ("
              << gflops   << " GFLOP/s)\n";

    mkl_free(A); mkl_free(B); mkl_free(C);
    return 0;
}

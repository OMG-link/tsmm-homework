#include "perf.h" // 仍然保留
#include <chrono>
#include <iostream>
#include <mkl.h>
#include <mkl_service.h>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "用法: " << argv[0] << " <m> <n> <k>\n";
        return 1;
    }
    const int m = std::stoi(argv[1]);
    const int n = std::stoi(argv[2]);
    const int k = std::stoi(argv[3]);

    std::cout << "A: " << m << 'x' << k << "  B: " << k << 'x' << n << "  C: " << m << 'x' << n << '\n';

    /* ---------- 分配 ---------- */
    double *A = (double *)mkl_malloc((size_t)m * k * sizeof(double), 64);
    double *B = (double *)mkl_malloc((size_t)k * n * sizeof(double), 64);
    double *C = (double *)mkl_malloc((size_t)m * n * sizeof(double), 64);
    if (!A || !B || !C) {
        std::cerr << "内存分配失败\n";
        return 2;
    }

#pragma omp parallel for
    for (long long i = 0; i < 1LL * m * k; ++i)
        A[i] = 1.0;
#pragma omp parallel for
    for (long long i = 0; i < 1LL * k * n; ++i)
        B[i] = 1.0;
#pragma omp parallel for
    for (long long i = 0; i < 1LL * m * n; ++i)
        C[i] = 0.0;

    mkl_set_num_threads(1);

    /* ---------- 热身 ---------- */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);

    /* ---------- 创建 & 清零计数器 ---------- */
    int fd_cycles = perf_event_cycles();
    int fd_instrs = perf_event_instructions();
    perf_reset(fd_cycles);
    perf_reset(fd_instrs);

    /* ---------- 正式计时 ---------- */
    auto t0 = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    auto t1 = std::chrono::high_resolution_clock::now();

    /* ---------- 读数并关闭 ---------- */
    uint64_t cycles = perf_read(fd_cycles);
    uint64_t instrs = perf_read(fd_instrs);
    perf_disable(fd_cycles);
    perf_disable(fd_instrs);
    perf_close_event(fd_cycles);
    perf_close_event(fd_instrs);

    /* ---------- 输出 ---------- */
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double sec = ms / 1000.0;
    double gflops = 2.0 * m * n * k / (sec * 1e9);
    double ipc = (double)instrs / cycles;

    std::cout << "C[0]        = " << C[0] << '\n'
              << "耗时         = " << ms << " ms  (" << gflops << " GFLOP/s)\n"
              << "cycles      = " << cycles << '\n'
              << "instructions = " << instrs << '\n'
              << "IPC         = " << ipc << '\n';

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return 0;
}
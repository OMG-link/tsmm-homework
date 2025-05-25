#pragma once

#include <chrono>
#include <functional>

#include "matrix.hpp"
#include "perf.h"

void test_kernel(const Matrix &a, const Matrix &b,
                 std::function<void(f64 *, const f64 *, const f64 *, size_t, size_t, size_t)> kernel,
                 const char *kernel_name) {
    using namespace std::chrono;

    Matrix c(a.rows(), b.cols());

    int fd_cycles = perf_event_cycles();
    int fd_instrs = perf_event_instructions();
    int fd_dtlb_m = perf_event_dtlb_miss();
    int fd_l1_a = perf_event_l1d_access();
    int fd_l1_m = perf_event_l1d_miss();

    perf_reset(fd_cycles);
    perf_reset(fd_instrs);
    perf_reset(fd_dtlb_m);
    perf_reset(fd_l1_a);
    perf_reset(fd_l1_m);

    auto start_opt = high_resolution_clock::now();
    kernel(c.data(), a.data(), b.data(), a.rows(), a.cols(), b.cols());
    auto end_opt = high_resolution_clock::now();

    perf_disable(fd_cycles);
    perf_disable(fd_instrs);
    perf_disable(fd_dtlb_m);
    perf_disable(fd_l1_a);
    perf_disable(fd_l1_m);

    uint64_t cycles = perf_read(fd_cycles);
    uint64_t instrs = perf_read(fd_instrs);
    uint64_t dtlb_m = perf_read(fd_dtlb_m);
    uint64_t l1_a = perf_read(fd_l1_a);
    uint64_t l1_m = perf_read(fd_l1_m);

    perf_close_event(fd_cycles);
    perf_close_event(fd_instrs);
    perf_close_event(fd_dtlb_m);
    perf_close_event(fd_l1_a);
    perf_close_event(fd_l1_m);

    auto seconds = duration_cast<duration<double>>(end_opt - start_opt).count();

    double M = static_cast<double>(a.rows());
    double N = static_cast<double>(b.cols());
    double K = static_cast<double>(a.cols());
    double operations = 2.0 * M * N * K;
    double gflops = operations / (seconds * 1e9);

    printf("%s | time = %.6e s | GFLOPS = %.2f\n", kernel_name, seconds, gflops);
    printf("cycles         = %lu\n", cycles);
    printf("instructions   = %lu\n", instrs);
    printf("- IPC          = %f\n", (double)instrs / cycles);
    printf("dTLB miss      = %lu\n", dtlb_m);
    printf("L1D  access    = %lu, miss = %lu\n", l1_a, l1_m);
}
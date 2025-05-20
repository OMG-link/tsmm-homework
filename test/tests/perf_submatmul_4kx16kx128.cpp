#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/matmul.h"
#include "matrix.hpp"
#include "perf.h"

void test(int m, int k, int n) {
    using namespace std::chrono;

    f64 *dst = (f64 *)malloc_aligned(m * n * sizeof(f64), 64);
    f64 *lhs = (f64 *)malloc_aligned(m * k * sizeof(f64), 64);
    f64 *rhs = (f64 *)malloc_aligned(k * n * sizeof(f64), 64);

    const int M_BLK = 128;
    const int K_BLK = 256;
    const int N_BLK = 128;

    int fd_cycles = perf_event_cycles();
    int fd_instrs = perf_event_instructions();
    int fd_clock = perf_event_task_clock();
    int fd_faults = perf_event_page_faults();
    int fd_dtlb_a = perf_event_dtlb_access();
    int fd_dtlb_m = perf_event_dtlb_miss();
    int fd_l1_a = perf_event_l1d_access();
    int fd_l1_m = perf_event_l1d_miss();
    int fd_llc_a = perf_event_llc_access();
    int fd_llc_m = perf_event_llc_miss();

    perf_reset(fd_cycles);
    perf_reset(fd_instrs);
    perf_reset(fd_clock);
    perf_reset(fd_faults);
    perf_reset(fd_dtlb_a);
    perf_reset(fd_dtlb_m);
    perf_reset(fd_l1_a);
    perf_reset(fd_l1_m);
    perf_reset(fd_llc_a);
    perf_reset(fd_llc_m);

    auto start_submatmul = high_resolution_clock::now();
    for (int m_idx = 0, m_block; m_idx < m; m_idx += m_block) {
        m_block = min_int(M_BLK, m - m_idx);
        for (int n_idx = 0, n_block; n_idx < n; n_idx += n_block) {
            n_block = min_int(N_BLK, n - n_idx);
            for (int k_idx = 0, k_block; k_idx < k; k_idx += k_block) {
                k_block = min_int(K_BLK, k - k_idx);
                const f64 *lhs_submat = lhs + m_idx * k + k_idx * m_block;
                const f64 *rhs_submat = rhs + n_idx * k + k_idx * n_block;
                f64 *dst_submat = dst + m_idx * n + n_idx;
                matmul_submat(dst_submat, lhs_submat, rhs_submat, m_block, k_block, n_block, n);
            }
        }
    }
    auto end_submatmul = high_resolution_clock::now();

    perf_disable(fd_cycles);
    perf_disable(fd_instrs);
    perf_disable(fd_clock);
    perf_disable(fd_faults);
    perf_disable(fd_dtlb_a);
    perf_disable(fd_dtlb_m);
    perf_disable(fd_l1_a);
    perf_disable(fd_l1_m);
    perf_disable(fd_llc_a);
    perf_disable(fd_llc_m);

    uint64_t cycles = perf_read(fd_cycles);
    uint64_t instrs = perf_read(fd_instrs);
    uint64_t clock = perf_read(fd_clock);
    uint64_t faults = perf_read(fd_faults);
    uint64_t dtlb_a = perf_read(fd_dtlb_a);
    uint64_t dtlb_m = perf_read(fd_dtlb_m);
    uint64_t l1_a = perf_read(fd_l1_a);
    uint64_t l1_m = perf_read(fd_l1_m);
    uint64_t llc_a = perf_read(fd_llc_a);
    uint64_t llc_m = perf_read(fd_llc_m);

    perf_close_event(fd_cycles);
    perf_close_event(fd_instrs);
    perf_close_event(fd_clock);
    perf_close_event(fd_faults);
    perf_close_event(fd_dtlb_a);
    perf_close_event(fd_dtlb_m);
    perf_close_event(fd_l1_a);
    perf_close_event(fd_l1_m);
    perf_close_event(fd_llc_a);
    perf_close_event(fd_llc_m);

    auto seconds = duration_cast<duration<double>>(end_submatmul - start_submatmul).count();

    double gflops = (2.0 * m * k * n) / (seconds * 1e9);
    printf("ok | submatmul = %.6f s | GFLOPS = %.2f\n", seconds, gflops);
    printf("task-clock     = %lu\n", clock);
    printf("cycles         = %lu\n", cycles);
    printf("instructions   = %lu\n", instrs);
    printf("- IPC          = %f\n", (double)instrs / cycles);
    printf("page-faults    = %lu\n", faults);
    printf("dTLB access    = %lu, miss = %lu\n", dtlb_a, dtlb_m);
    printf("L1D  access    = %lu, miss = %lu\n", l1_a, l1_m);
    printf("LLC  access    = %lu, miss = %lu\n", llc_a, llc_m);

    free(lhs);
    free(rhs);
    free(dst);
}

int main() { test(4000, 16000, 128); }
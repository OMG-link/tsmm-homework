#include <iostream>
#include <cstdlib>
#include <chrono>

#include "core/matmul.h"      // matmul_block, f64, malloc_aligned …
#include "core/pack.h"
#include "perf.h"             // --- NEW: perf_event_* helpers

// --------------------- 根据你的内核生成时的块大小调整 ---------------------
static constexpr int M_BLK = 128;   
static constexpr int K_BLK = 256;
static constexpr int N_BLK = 120;  
// -------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "用法: " << argv[0] << " <m> <k> <n>\n";
        return EXIT_FAILURE;
    }

    const int m = std::atoi(argv[1]);
    const int k = std::atoi(argv[2]);
    const int n = std::atoi(argv[3]);

    /* ---------- 128 B 对齐分配（与内核一致） ---------- */
    f64* lhs = static_cast<f64*>(malloc_aligned(static_cast<size_t>(m) * k * sizeof(f64), 128));
    f64* rhs = static_cast<f64*>(malloc_aligned(static_cast<size_t>(k) * n * sizeof(f64), 128));
    f64* dst = static_cast<f64*>(malloc_aligned(static_cast<size_t>(m) * n * sizeof(f64), 128));
    if (!lhs || !rhs || !dst) {
        std::cerr << "内存分配失败\n";
        return EXIT_FAILURE;
    }

    /* ---------- 简单可重复的初始化 ---------- */
    for (int i = 0; i < m * k; ++i) lhs[i] = static_cast<f64>((i % 97) - 48.0);
    for (int i = 0; i < k * n; ++i) rhs[i] = static_cast<f64>(((i * 7) % 89) - 44.0);
    std::fill(dst, dst + static_cast<size_t>(m) * n, 0.0);

    /* ---------- 热身 ---------- */
    matmul_block(dst, lhs, rhs, m, k, n, M_BLK, K_BLK, N_BLK);

    /* ---------- 创建 & 清零计数器 ---------- */
    int fd_cycles = perf_event_cycles();         // --- NEW
    int fd_instrs = perf_event_instructions();   // --- NEW
    perf_reset(fd_cycles);                       // --- NEW
    perf_reset(fd_instrs);                       // --- NEW

    /* ---------- 正式计时 ---------- */
    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_block(dst, lhs, rhs, m, k, n, M_BLK, K_BLK, N_BLK);
    auto t1 = std::chrono::high_resolution_clock::now();

    /* ---------- 读数并关闭 ---------- */
    uint64_t cycles = perf_read(fd_cycles);      // --- NEW
    uint64_t instrs = perf_read(fd_instrs);      // --- NEW
    perf_disable(fd_cycles);                     // --- NEW
    perf_disable(fd_instrs);                     // --- NEW
    perf_close_event(fd_cycles);                 // --- NEW
    perf_close_event(fd_instrs);                 // --- NEW

    /* ---------- 结果计算 ---------- */
    double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double sec = ms / 1000.0;
    double gflops = 2.0 * m * n * k / (sec * 1e9);   // 2·m·n·k / time
    double ipc    = static_cast<double>(instrs) / cycles;

    /* ---------- 打印 ---------- */
    std::cout << "C[0]        = " << dst[0]      << '\n'
              << "耗时         = " << ms         << " ms  (" << gflops << " GFLOP/s)\n"
              << "cycles      = " << cycles      << '\n'
              << "instructions = " << instrs     << '\n'
              << "IPC         = " << ipc         << '\n';

    free(lhs);
    free(rhs);
    free(dst);
    return 0;
}

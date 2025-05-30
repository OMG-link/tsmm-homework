#include "core/kernels/8x16000x16.hpp"
#include "matmul.hpp"
#include "test.h"

bool test() {
    MatMul8x16000x16 kernel;
    const int m = 8;
    const int k = 16000;
    const int n = 16;
    Matrix a = Matrix::from_file("./test/data/8x16000_1.txt");
    Matrix b = Matrix::from_file("./test/data/16000x16_1.txt");
    Matrix c(m, n);
    kernel.compute(c.data(), a.data(), b.data(), m, k, n);
    Matrix ans = matmul_generic(a, b);
    return c == ans;
}

int main() { check(test()); }
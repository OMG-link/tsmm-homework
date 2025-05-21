#include "core/kernels/4000x128x16000.hpp"
#include "matmul.hpp"
#include "test.h"

bool test() {
    MatMul4000x128x16000 kernel;
    const int m = 4000;
    const int k = 128;
    const int n = 16000;
    Matrix a = Matrix::from_file("./test/data/4000x128_1.txt");
    Matrix b = Matrix::from_file("./test/data/128x16000_1.txt");
    Matrix c(m, n);
    kernel.compute(c.data(), a.data(), b.data(), m, k, n);
    Matrix ans = matmul_generic(a, b);
    return c == ans;
}

int main() { check(test()); }
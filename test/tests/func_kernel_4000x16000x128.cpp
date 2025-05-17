#include "core/kernels/4000x16000x128.hpp"
#include "matmul.hpp"
#include "test.h"

bool test() {
    MatMul4000x16000x128 kernel;
    const int m = 4000;
    const int k = 16000;
    const int n = 128;
    Matrix a = Matrix::from_file("./test/data/4000x16000_1.txt");
    Matrix b = Matrix::from_file("./test/data/16000x128_1.txt");
    Matrix c(m, n);
    kernel.compute(c.data(), a.data(), b.data(), m, k, n);
    Matrix ans = matmul_generic(a, b);
    return c == ans;
}

int main() { check(test()); }
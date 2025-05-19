#include "core/kernels/144x144x144.hpp"
#include "matmul.hpp"
#include "test.h"

bool test() {
    MatMul144x144x144 kernel;
    const int m = 144;
    const int k = 144;
    const int n = 144;
    Matrix a = Matrix::from_file("./test/data/144x144_1.txt");
    Matrix b = Matrix::from_file("./test/data/144x144_1.txt");
    Matrix c(m, n);
    kernel.compute(c.data(), a.data(), b.data(), m, k, n);
    Matrix ans = matmul_generic(a, b);
    return c == ans;
}

int main() { check(test()); }
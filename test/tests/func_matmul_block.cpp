#include "core/matmul.h"
#include "matmul.hpp"
#include "test.h"

#define def_test(in_n, in_k, in_m)                                                                                     \
    bool test_##in_n##x##in_k##x##in_m() {                                                                             \
        Matrix a = Matrix::from_file("./test/data/" #in_n "x" #in_k "_1.txt");                                         \
        Matrix b = Matrix::from_file("./test/data/" #in_k "x" #in_m "_1.txt");                                         \
        Matrix c = matmul_block(a, b);                                                                                 \
        Matrix ans = matmul_generic(a, b);                                                                             \
        return c == ans;                                                                                               \
    }

#define run_test(in_m, in_k, in_n) check(test_##in_m##x##in_k##x##in_n())

Matrix matmul_block(const Matrix &lhs, const Matrix &rhs) {
    size_t m = lhs.rows();
    size_t k = lhs.cols();
    size_t n = rhs.cols();
    Matrix dst(m, n);
    matmul_block(dst.data(), lhs.data(), rhs.data(), m, k, n, 64, 64, 64);
    return dst;
}

def_test(16, 16, 16);
def_test(200, 400, 300);

int main() {
    run_test(16, 16, 16);
    run_test(200, 400, 300);
}
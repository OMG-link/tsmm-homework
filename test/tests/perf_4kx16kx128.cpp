#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "matmul.hpp"

void test_and_verify(const Matrix &a, const Matrix &b) {
    using namespace std::chrono;

    // Timer - optimized
    auto start_opt = high_resolution_clock::now();
    Matrix c = matmul(a, b);
    auto end_opt = high_resolution_clock::now();
    auto duration_opt = duration_cast<milliseconds>(end_opt - start_opt).count();

    // Timer - generic
    auto start_gen = high_resolution_clock::now();
    Matrix ans = matmul_generic(a, b);
    auto end_gen = high_resolution_clock::now();
    auto duration_gen = duration_cast<milliseconds>(end_gen - start_gen).count();

    if (c == ans) {
        printf("ok | optimized timer = %ldms, generic timer = %ldms\n", duration_opt, duration_gen);
    } else {
        puts("wa | result is wrong");
    }
}

int main() {
    Matrix a = Matrix::from_file("./test/data/4000x16000_1.txt");
    Matrix b = Matrix::from_file("./test/data/16000x128_1.txt");
    test_and_verify(a, b);
}
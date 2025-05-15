#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "matmul.hpp"

void test_and_verify(const Matrix &a, const Matrix &b) {
    using namespace std::chrono;

    auto start_opt = high_resolution_clock::now();
    Matrix c = matmul(a, b);
    auto end_opt = high_resolution_clock::now();
    auto duration_opt = duration_cast<milliseconds>(end_opt - start_opt).count();

    printf("ok | optimized timer = %ldms\n", duration_opt);
}

int main() {
    Matrix a(4000, 16000), b(16000, 128);
    test_and_verify(a, b);
}
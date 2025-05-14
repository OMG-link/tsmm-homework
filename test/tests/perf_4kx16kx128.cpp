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

    printf("ok | optimized timer = %ldms\n", duration_opt);
}

int main() {
    Matrix a = Matrix::from_file("./test/data/4000x16000_1.txt");
    Matrix b = Matrix::from_file("./test/data/16000x128_1.txt");
    test_and_verify(a, b);
}
#include "core/pack.h"
#include "test.h"

#define def_test(in_n, in_m, blk_n, blk_m, fty, OPK_SIZE)                                                              \
    bool test_##in_n##x##in_m##_##blk_n##x##blk_m##_##fty() {                                                          \
        size_t output_size = sizeof(output_##in_n##x##in_m##_##blk_n##x##blk_m##_##fty);                               \
        f64 *dst = (f64 *)malloc(output_size);                                                                         \
        pack_matrix_##fty(dst, input_##in_n##x##in_m, in_n, in_m, blk_n, blk_m, OPK_SIZE);                             \
        bool ok = memcmp(dst, output_##in_n##x##in_m##_##blk_n##x##blk_m##_##fty, output_size) == 0;                   \
        free(dst);                                                                                                     \
        return ok;                                                                                                     \
    }

#include "../data/pack.inc"

def_test(5, 7, 3, 5, lhs, 8);
def_test(16, 16, 8, 8, lhs, 8);
def_test(16, 16, 8, 16, lhs, 8);
def_test(16, 16, 16, 8, lhs, 8);
def_test(16, 16, 16, 16, lhs, 8);
def_test(5, 7, 3, 5, rhs, 16);
def_test(16, 16, 8, 8, rhs, 16);
def_test(16, 16, 8, 16, rhs, 16);
def_test(16, 16, 16, 8, rhs, 16);
def_test(16, 16, 16, 16, rhs, 16);

int main() {
    check(test_5x7_3x5_lhs());
    check(test_16x16_8x8_lhs());
    check(test_16x16_8x16_lhs());
    check(test_16x16_16x8_lhs());
    check(test_16x16_16x16_lhs());
    check(test_5x7_3x5_rhs());
    check(test_16x16_8x8_rhs());
    check(test_16x16_8x16_rhs());
    check(test_16x16_16x8_rhs());
    check(test_16x16_16x16_rhs());
}

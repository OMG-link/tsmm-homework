#include "utils.h"

static const int DST_M_BLK = VLEN / sizeof(f64);     // must be a multiple of VLEN / sizeof(f64)
static const int DST_N_BLK = 2 * VLEN / sizeof(f64); // must be a multiple of VLEN / sizeof(f64)

#ifdef __cplusplus
extern "C" {
#endif

void pack_matrix_lhs(f64 *dst, const f64 *src, int m, int k, const int M_BLK, const int K_BLK);
void pack_matrix_rhs(f64 *dst, const f64 *src, int k, int n, const int K_BLK, const int N_BLK);

#ifdef __cplusplus
}
#endif

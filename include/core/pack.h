#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void pack_matrix_lhs(f64 *RESTRICT dst, const f64 *RESTRICT src, int m, int k, const int M_BLK, const int K_BLK,
                     const int OPK_M_BLK);
void pack_matrix_rhs(f64 *RESTRICT dst, const f64 *RESTRICT src, int k, int n, const int K_BLK, const int N_BLK,
                     const int OPK_N_BLK);

#ifdef __cplusplus
}
#endif

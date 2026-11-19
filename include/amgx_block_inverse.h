// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "amgx_c.h"

#ifdef __cplusplus
extern "C" {
#endif
/** Optional dDDI device block-Jacobi inverse supplied by an application.
 * On every setup/resetup (including coarse levels), diag[i] indexes a
 * row-major block in values. Fill the corresponding packed inverse row.
 * Run on the current device/default stream and finish before returning;
 * return zero on success. Never retain buffers or throw through this ABI.
 * The descriptor must outlive all solver/config copies that reference it.
 * Set it before solver creation. A NULL descriptor selects the native inverse.
 */
typedef struct {
    int (*compute)(int rows, int width, const int *diag, const double *values, double *inverse);
} AMGX_block_inverse;
AMGX_RC AMGX_API AMGX_config_set_block_inverse(AMGX_config_handle cfg, const AMGX_block_inverse *provider);
#ifdef __cplusplus
}
#endif

// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <matrix.h>
#include <cuda_runtime.h>

namespace amgx
{
namespace aggregation
{

typedef Matrix<typename TemplateMode<AMGX_mode_dDDI>::Type> GeometryMatrix;
void attachGeometricStrength(GeometryMatrix &A, const double *device_weights);
void loadGeometricStrength(const GeometryMatrix &A, float *selector_weights, cudaStream_t stream);
void copyGeometricStrength(const GeometryMatrix &A, double *host_weights, cudaStream_t stream);
void coarsenGeometricStrength(const GeometryMatrix &A, GeometryMatrix &Ac, const int *aggregates,
                              const int *restriction_rows, const int *restriction_cols);
bool refreshRetainedCoarseValues(const GeometryMatrix &A, GeometryMatrix &Ac, const int *aggregates,
                                 const int *restriction_rows, const int *restriction_cols);

template<class TConfig>
void loadGeometricStrength(const Matrix<TConfig> &, float *, cudaStream_t)
{
    FatalError("Geometric aggregation requires single-device dDDI", AMGX_ERR_BAD_PARAMETERS);
}
template<class TConfig>
void coarsenGeometricStrength(const Matrix<TConfig> &, Matrix<TConfig> &, const int *, const int *, const int *)
{
    FatalError("Geometric aggregation requires single-device dDDI", AMGX_ERR_BAD_PARAMETERS);
}
template<class TConfig>
bool refreshRetainedCoarseValues(const Matrix<TConfig> &, Matrix<TConfig> &, const int *, const int *, const int *)
{
    return false;
}

}
}

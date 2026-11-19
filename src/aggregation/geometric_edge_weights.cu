// SPDX-License-Identifier: BSD-3-Clause
#include <aggregation/geometric_edge_weights.h>
#include <thrust_wrapper.h>
#include <cmath>
#include <climits>

namespace amgx
{
namespace aggregation
{

// Keep keys with FP64 strengths. A matrix may reorder its own columns during setup.
struct GeometricEdgeWeights
{
    GeometryMatrix::IVector rows, cols;
    Vector<GeometryMatrix::TConfig> weights;

    explicit GeometricEdgeWeights(const GeometryMatrix &A) :
        rows(A.row_offsets), cols(A.col_indices), weights(A.get_num_nz()) {}
};

namespace
{
#include <aggregation/selectors/edge_weight_perturbation.h>

__device__ double selectorWeight(double value, int, int) { return value; }
__device__ float selectorWeight(float value, int row, int col)
{
    float delta = scaling_factor<float>() * hash_val(min(row, col), max(row, col)) / float(UINT_MAX);
    return value + delta * value;
}

__device__ int columnSlot(const int *cols, int first, int last, int col)
{
    for (int k = first; k < last; ++k)
        if (cols[k] == col) return k;
    return -1;
}

const int threads = 128;

void checkedSetup(cudaStream_t stream, const GeometryMatrix::IVector &failure)
{
    int invalid = 0;
    cudaError_t e = cudaMemcpyAsync(&invalid, failure.raw(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) FatalError(cudaGetErrorString(e), AMGX_ERR_INTERNAL);
    if (invalid)
        FatalError("Invalid aggregation strength or incompatible block CSR topology", AMGX_ERR_BAD_PARAMETERS);
}

// Materialize scalar edges in the active physical CSR order. Explicit diagonal
// and separate-diagonal storage both yield zero diagonal weight. The common
// case has identical column ordering and requires one comparison per edge.
template<class Output>
__global__ void gatherStrength(int n, const int *rows, const int *cols, const int *saved_rows,
                               const int *saved_cols, const double *saved, Output *out, int *failure)
{
    int row = blockIdx.x;
    if (row >= n) return;
    const int first = saved_rows[row], last = saved_rows[row + 1];
    for (int k = rows[row] + threadIdx.x; k < rows[row + 1]; k += blockDim.x)
    {
        const int col = cols[k];
        if (col == row) { out[k] = 0; continue; }
        const int slot = (k >= first && k < last && saved_cols[k] == col)
            ? k : columnSlot(saved_cols, first, last, col);
        const Output value = slot < 0 ? Output(0) : selectorWeight(static_cast<Output>(saved[slot]), row, col);
        if (!(value > 0) || !isfinite(value) || col < 0 || col >= n)
            atomicExch(failure, 1);
        out[k] = value;
    }
}

template<class Output>
void readStrength(const GeometryMatrix &A, Output *out, cudaStream_t stream)
{
    if (!A.geometric_edge_weights || !A.is_matrix_singleGPU())
        FatalError("weight_formula=2 requires attached single-device geometric strengths", AMGX_ERR_BAD_PARAMETERS);
    const GeometricEdgeWeights &g = *A.geometric_edge_weights;
    if (g.rows.size() != size_t(A.get_num_rows()) + 1 || g.cols.size() < g.weights.size())
        FatalError("Geometric graph no longer matches matrix rows", AMGX_ERR_BAD_PARAMETERS);
    GeometryMatrix::IVector failure(1, 0);
    if (A.get_num_rows())
    {
        gatherStrength<<<A.get_num_rows(), threads, 0, stream>>>(A.get_num_rows(), A.row_offsets.raw(),
            A.col_indices.raw(), g.rows.raw(), g.cols.raw(), g.weights.raw(), out, failure.raw());
        cudaCheckError();
    }
    checkedSetup(stream, failure);
}

// R lists the fine rows of a coarse row. Each coarse edge owns one reduction,
// so multiple fine faces between a pair accumulate in a reproducible order.
__global__ void sumCrossingEdges(int n, const int *cr, const int *cc, double *coarse,
                                 const int *rr, const int *rc, const int *group,
                                 const int *fr, const int *fc, const double *fine, int *failure)
{
    const int i = blockIdx.x;
    for (int k = cr[i] + threadIdx.x; k < cr[i + 1]; k += blockDim.x)
    {
        const int j = cc[k];
        double sum = 0;
        if (i != j)
        {
            for (int r = rr[i]; r < rr[i + 1]; ++r)
                for (int e = fr[rc[r]]; e < fr[rc[r] + 1]; ++e)
                    if (group[fc[e]] == j) sum += fine[e];
            if (!(sum > 0) || !isfinite(sum) || j < 0 || j >= n)
                atomicExch(failure, 1);
        }
        coarse[k] = sum;
    }
}

// One CTA owns a coarse block row, one lane owns a matrix component. Traverse
// the actual retained CSR; no hash, pattern rebuild, floating-point atomic, or
// temporary matrix. Both layouts accumulate intra-aggregate couplings onto
// the coarse diagonal; a separate fine diagonal contributes exactly once.
__global__ void refreshRAP(int b, const int *fr, const int *fc, const int *fd, const double *av,
                           const int *cr, const int *cc, const int *cd, double *cv,
                           const int *rr, const int *rc, const int *group, int *failure)
{
    const int i = blockIdx.x, ncomp = b*b;
    const int first = cr[i], last = cr[i+1];
    for (int k = first; k < last; ++k)
        for (int c = threadIdx.x; c < ncomp; c += blockDim.x) cv[ncomp*k + c] = 0;
    if (cd)
        for (int c = threadIdx.x; c < ncomp; c += blockDim.x) cv[ncomp*cd[i] + c] = 0;
    for (int p = rr[i]; p < rr[i+1]; ++p)
    {
        const int f = rc[p];
        if (fd)
            for (int c = threadIdx.x; c < ncomp; c += blockDim.x)
                cv[ncomp*cd[i] + c] += av[ncomp*fd[f] + c];
        for (int e = fr[f]; e < fr[f+1]; ++e)
        {
            const int j = group[fc[e]];
            if (j < 0) continue;
            const int k = (cd && i == j) ? cd[i] : columnSlot(cc, first, last, j);
            if (k < 0) { if (threadIdx.x == 0) atomicExch(failure, 1); continue; }
            for (int c = threadIdx.x; c < ncomp; c += blockDim.x)
                cv[ncomp*k + c] += av[ncomp*e + c];
        }
    }
}
}

void attachGeometricStrength(GeometryMatrix &A, const double *device_weights)
{
    auto g = std::make_shared<GeometricEdgeWeights>(A);
    const cudaStream_t stream = amgx::thrust::global_thread_handle::get_stream();
    // Validate and copy into the finished snapshot before replacing an existing attachment.
    GeometryMatrix::IVector failure(1, 0);
    if (A.get_num_rows())
    {
        gatherStrength<<<A.get_num_rows(), threads, 0, stream>>>(A.get_num_rows(), A.row_offsets.raw(),
            A.col_indices.raw(), A.row_offsets.raw(), A.col_indices.raw(), device_weights,
            g->weights.raw(), failure.raw());
        cudaCheckError();
    }
    checkedSetup(stream, failure);
    A.geometric_edge_weights = g;
}

void loadGeometricStrength(const GeometryMatrix &A, float *out, cudaStream_t stream) { readStrength(A, out, stream); }
void copyGeometricStrength(const GeometryMatrix &A, double *host_out, cudaStream_t stream)
{
    Vector<GeometryMatrix::TConfig> current(A.get_num_nz());
    readStrength(A, current.raw(), stream);
    if (current.size())
    {
        cudaError_t e = cudaMemcpyAsync(host_out, current.raw(), current.size() * sizeof(double), cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) FatalError(cudaGetErrorString(e), AMGX_ERR_INTERNAL);
    }
}

void coarsenGeometricStrength(const GeometryMatrix &A, GeometryMatrix &Ac, const int *groups,
                              const int *rr, const int *rc)
{
    auto g = std::make_shared<GeometricEdgeWeights>(Ac);
    const auto &fine = *A.geometric_edge_weights;
    const cudaStream_t stream = amgx::thrust::global_thread_handle::get_stream();
    GeometryMatrix::IVector failure(1, 0);
    if (Ac.get_num_rows())
    {
        sumCrossingEdges<<<Ac.get_num_rows(), threads, 0, stream>>>(Ac.get_num_rows(), g->rows.raw(),
            g->cols.raw(), g->weights.raw(), rr, rc, groups, fine.rows.raw(), fine.cols.raw(),
            fine.weights.raw(), failure.raw());
        cudaCheckError();
    }
    checkedSetup(stream, failure);
    Ac.geometric_edge_weights = g;
}

bool refreshRetainedCoarseValues(const GeometryMatrix &A, GeometryMatrix &Ac, const int *groups,
                                 const int *rr, const int *rc)
{
    if (!A.is_matrix_singleGPU() || A.get_block_dimy() != Ac.get_block_dimy() ||
        A.hasProps(DIAG) != Ac.hasProps(DIAG)) return false;
    const cudaStream_t stream = amgx::thrust::global_thread_handle::get_stream();
    GeometryMatrix::IVector failure(1, 0);
    if (Ac.get_num_rows())
    {
        refreshRAP<<<Ac.get_num_rows(), threads, 0, stream>>>(A.get_block_dimy(), A.row_offsets.raw(),
            A.col_indices.raw(), A.hasProps(DIAG) ? A.diag.raw() : nullptr, A.values.raw(),
            Ac.row_offsets.raw(), Ac.col_indices.raw(), Ac.hasProps(DIAG) ? Ac.diag.raw() : nullptr,
            Ac.values.raw(), rr, rc, groups, failure.raw());
        cudaCheckError();
    }
    checkedSetup(stream, failure);
    return true;
}
}
}

# Native block widths, attached device values and allocation diagnostics

This extension builds on `codex/native-blocks-6-7` and the immutable upstream
v2.5.0 revision `cc1cebdbb32b14d33762d4ddabcb2e23c1669f47`.
Source changes are complete; grouped validation is required before promotion.
The previous validation in `NativeBlockFixes.md` applies to that older patch,
not automatically to this extension.

## Selected block catalog

The selected single-device dDDI `AGGREGATION/SIZE_2/LOW_DEG/BLOCK_JACOBI`
path supports native square widths 1,2,3,4,5,6,7,8,9,19, retaining width 10 as
an existing reference. Widths 9 and 19 have explicit restriction and LOW_DEG
coarse-construction dispatch. Every 32-entry matrix tile is retained, including
separate diagonals. The independent host/GPU Galerkin test includes all these
widths, both diagonal forms and rows crossing the 32-neighbor boundary.

Large-block diagonal inversion chooses its cooperative launch size so dynamic
shared memory remains below 48 KiB. Scratch is allocated for the actual thread
count. Small-block inverse phases and the four-component smoother explicitly
synchronize participating lanes; volatile shared storage alone does not order
independently scheduled lanes. The original provider inversion/epsilon policy
is retained. This extension does not replace it with a filtered SVD or imply
that a singular unprojected reconstruction operator is equivalent to its
pseudoinverse-projected fixed point.

## Experimental native numeric attachment

`AMGX_matrix_attach_device_values` accepts only the selected device's compact
row-major FP64 numeric buffer in dDDI mode. CSR indices are uploaded normally.
For `separate_diagonal=1` the buffer is `[nnz off-diagonal blocks][n diagonal
blocks]`; otherwise it is `[nnz inline-diagonal blocks][one zero sentinel
block]`. `value_count` must match this physical extent exactly. The caller
owns and retains that allocation, serializes all use and keeps it read-only
during setup/solve. Values may change between solves only with a new setup.

The device Vector's restricted external overlay keeps its underlying Thrust
allocation empty, provides compatible raw/iterator access and never frees the
foreign pointer. Copy construction creates owned values through those
iterators. Fill, size-changing resize and swap are rejected. The empty
matrix's private one-scalar sentinel is retired before attachment.

This overlay is restricted provider-internal storage, not a general replacement
for the publicly inherited Thrust container. Direct base-container operations
(`assign`, `insert`, `erase`, or an explicit base-class upcast) are unsupported
for an attached value vector. They do not enter the selected profile and are
not part of the native C API. New algorithms must be audited for these paths
before being admitted; only the guarded Vector operations above are supported.

The selected setup rejects coloring, reordering, inserted-diagonal conversion
and equation scaling for an attached payload. Public upload/replace cannot
write through an attachment. Neither attachment nor destruction takes ownership
of the caller's allocation. This is a synchronous provider preparation/apply
route; the existing public API still has default-stream operations and joins.
It is **not** caller-stream asynchronous or CUDA-graph qualified.

The downstream managed adapter additionally retains a Runtime Array handle,
acquires leases during calls and checks mutation epochs before use. A raw C
API caller must supply the same lifetime/readiness guarantees itself.

## Memory reporting

`AMGX_solver_get_hierarchy_memory` reports per-level capacities for owned or
borrowed numeric data, symbolic arrays, Jacobi inverse, aggregate/transfer maps
and cycle/smoother workspace. Borrowed finest values are explicitly separate
from provider ownership. No device pointers escape through this query.

`AMGX_get_allocator_memory` reports optional logical allocator live/peak
requests (`AMGX_TRACK_ALLOCATIONS=1` before the first resource), custom pool
used/reserved bytes and CUDA default-pool current/high values. The default
CUDA pool is process/device scoped and can be shared with another library;
do not label all its reserve as AmgX-owned or add it to live level capacities.
Logical allocation retirement occurs when free is submitted, so its peak is
not a physical pending-free peak. Query after joining provider work. CUDA
library-private allocations are outside the AmgX request ledger.

## Focused qualification commands

Run only in an explicitly allocated idle device context:

```sh
cmake --build <build> --target amgx_tests_launcher -j16
CUDA_VISIBLE_DEVICES=0 <build>/src/amgx_tests_launcher --mode dDDI AggregatesCoarseGeneratorTest
CUDA_VISIBLE_DEVICES=0 <build>/src/amgx_tests_launcher --mode dDDI NativeValuesStorageTest
```

The downstream tests add all requested widths, inverse exact-solve tails,
public/attached equivalence, value refresh, retained allocation lifetime,
unchanged input values, stale-epoch rejection and native flow matrix aliases.
These planned checks are not a result until the execution ledger records them.

## Geometry-only aggregation with full structure reuse

For single-device dDDI block Jacobi, supply one FP64 edge strength in each fine
CSR slot with `AMGX_matrix_attach_edge_weights` after uploading or attaching
the matrix and before solver setup. The matrix retains its own copy. In the
AMG config select `weight_formula=2`, a SIZE_2, SIZE_4 or SIZE_8 selector,
LOW_DEG and `structure_reuse_levels=-1`. An inline diagonal has weight zero.
At each level the next graph sums all conductances crossing between the same
two aggregates. The matching perturbation applies only to the temporary float
selector weights.

After updating physical coefficients in place (or replacing uploaded values)
call `AMGX_solver_resetup` on the same matrix. This preserves the CSR patterns,
aggregate membership and transfer maps, recomputes the full physical blocks
of the Galerkin product and refreshes the Jacobi smoother inverses. Callers
create a new matrix and hierarchy when the graph or geometric strengths change.
The diagnostic `examples/validate_geometry_hierarchy.py` compares each layer
against CPU edge and matrix sums and checks a refreshed cycle against fresh setup.

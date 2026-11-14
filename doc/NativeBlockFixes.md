# Native 6x6 and 7x7 aggregation blocks

This branch is based on upstream AmgX v2.5.0, commit
`cc1cebdbb32b14d33762d4ddabcb2e23c1669f47`. It contains a focused correction
for single-device aggregation AMG with native dense square blocks. It does
not scalarize or pad the matrices.

## Qualified path

The exercised path uses `dDDI`, `AGGREGATION`, `SIZE_2`, `LOW_DEG`,
`BLOCK_JACOBI`, a V-cycle and `NOSOLVER` at the coarsest level. Native block
sizes 5, 6 and 7 were exercised through complete preconditioner applications.
Other smoothers and coarse solvers have their own supported block sizes;
adding these dispatches is not a claim of universal block-size support.

## Corrections

1. Add block-size 6 and 7 dispatches to LOW_DEG coarse-matrix construction and
   residual restriction.
2. In the large-block LOW_DEG kernel, advance and shuffle the 32 column indices
   actually loaded by a warp. Iterate over matrix-entry tiles independently.
   Preserve all separate-diagonal tiles without reducing unrelated entries.
3. Allocate generic block-Jacobi inverse scratch for the launched thread count,
   rather than the grid-limit count. The previous allocation could be exceeded
   beyond 256 block rows. Correct the Schur update to `A_ij - A_ik*A_kj` and
   explicitly synchronize the 16-lane shared-memory subgroups.
4. Strengthen the independent host/GPU Galerkin oracle: include sizes 6/7/8/10,
   more than 32 neighbors, inline and separate diagonals, and every coefficient.
   The old separate-diagonal comparison read the first matrix on both sides.
   Include this test in the explicit `amgx_tests_launcher` target.

## Validation

The exact six-file source patch on this branch was tested with CUDA 12.6,
GCC 12, and an A100 before publication. The upstream Galerkin oracle passed
in `dDDI` mode, including Compute Sanitizer memcheck with zero reported errors:

```sh
cmake --build <build> --target amgx_tests_launcher -j16
<build>/src/amgx_tests_launcher --mode dDDI AggregatesCoarseGeneratorTest
compute-sanitizer --tool memcheck --error-exitcode=99 \
  <build>/src/amgx_tests_launcher --mode dDDI AggregatesCoarseGeneratorTest
```

Separate downstream integration tests (not bundled here) passed 67,605
assertions and memcheck with zero reported errors. They cover native 5/6/7
multilevel contraction, value refresh, multiple live provider owners, and
native 6/7 nonsymmetric block-diagonal exact solves with 1, 63 and 5,003 rows.
The exact-solve test sets the **smoother's** relaxation factor to 1; a factor
on the enclosing AMG object does not configure a separately scoped smoother.

The tested patch SHA-256 is
`d94fa872ed1210cd17a7c5bba9b9aaedd47908b81ee22ab151df4bf256506de6`;
the resulting shared-library SHA-256 is
`b4696951aab6ad794bada03ea3b9d341d8dfd4b640d7ae6bc7dc453943ea3d81`.
Build hashes describe the tested artifacts, not reproducible-build guarantees.

## Scope limitations

This correction does not introduce a caller-selected CUDA stream API or promise
CUDA graph capture. Existing internal stream handling, default-stream launches,
host synchronization and library-handle ownership still require a separate
audit before asynchronous embedding.

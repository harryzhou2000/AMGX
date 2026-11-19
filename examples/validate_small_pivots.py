#!/usr/bin/env python3
"""Verify both products of the actual FP64 block-Jacobi inverse on scaled 6×6 diagonals."""
import argparse
import ctypes as ct
import json
from contextlib import ExitStack

from validate_geometry_hierarchy import API, DDI, F, I, array

# SUBOFF ACM–SST first nonlinear update, finest row 1382538, row-major.
BLOCK = (
    1.0982265895729199e-07, 2.0988888451910419e-10, 5.7224447982924124e-12, 2.7036305884416233e-08, 0, 0,
    1.6767291918481589e-10, 9.9993819379800899e-08, -3.8436218628002946e-11, 4.2215970917483704e-11, 0, 0,
    4.6831975112963931e-12, -3.843621853770013e-11, 1.0002505376973587e-07, 1.039253633272404e-12, 0, 0,
    -1.2364439713096945e-09, -2.7900718670310992e-11, -4.4390443790038015e-13, 2.5007275959791394e-08, 0, 0,
    -3.7402430138393803e-19, -8.439967521151394e-21, -1.3428096962188171e-22, 3.1951240329257296e-19, 9.6044867933740949e-08, 0,
    -8.9765832348515185e-10, -2.0255921346574205e-11, -3.2227564696887922e-13, 7.6682976653262404e-10, 0, 9.6044650059429921e-08,
)
SCALES = (1.e-4, 1., 100.)


def verify(api, device):
    # Six adjacent independent blocks per scale. Supply one basis vector per
    # block to recover all columns of Dinv from a single Jacobi application.
    rows = len(SCALES) * 6
    blocks = [scale*v for scale in SCALES for j in range(6) for v in BLOCK]
    rhs = [float(i == j) for scale in SCALES for j in range(6) for i in range(6)]
    config = {"config_version": 2, "solver": {"solver": "BLOCK_JACOBI", "max_iters": 1,
              "relaxation_factor": 1., "monitor_residual": 0, "print_solve_stats": 0}}
    with ExitStack() as stack:
        cfg = api.own(stack, "config", json.dumps(config).encode())
        resources = api.own(stack, "resources", cfg, None, 1, ct.byref(I(device)))
        matrix = api.own(stack, "matrix", resources, DDI)
        api.matrix_upload_all(matrix, rows, rows, 6, 6, array(I, range(rows+1)),
                              array(I, range(rows)), array(F, blocks), None)
        solver = api.own(stack, "solver", resources, DDI, cfg)
        api.solver_setup(solver, matrix)
        b = api.own(stack, "vector", resources, DDI)
        x = api.own(stack, "vector", resources, DDI)
        for vec in (b, x):
            api.vector_bind(vec, matrix)
        api.vector_upload(b, rows, 6, array(F, rhs))
        api.vector_set_zero(x, rows, 6)
        api.solver_solve_with_0_initial_guess(solver, b, x)
        solution = (F * len(rhs))()
        api.vector_download(x, solution)

    failed_scales = 0
    for k, scale in enumerate(SCALES):
        d = [v*scale for v in BLOCK]
        inv = [solution[k*36+j*6+i] for i in range(6) for j in range(6)]
        def defect(left, right):
            return max(sum(abs(sum(left[6*i+m]*right[6*m+j] for m in range(6)) - (i==j))
                           for j in range(6)) for i in range(6))
        errors = [defect(inv, d), defect(d, inv)]
        print(json.dumps({"scale": scale, "left_defect": errors[0], "right_defect": errors[1]}))
        if not all(v < 3.e-13 for v in errors):
            failed_scales += 1
    if failed_scales:
        raise ValueError(f"Incorrect inverse at {failed_scales} scales")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amgx", default="libamgxsh.so")
    parser.add_argument("--cudart", default="libcudart.so")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    api = API(args.amgx, args.cudart)
    api.set_device(args.device)
    with ExitStack() as stack:
        api.initialize()
        stack.callback(api.finalize)
        api.initialize_plugins()
        stack.callback(api.finalize_plugins)
        verify(api, args.device)

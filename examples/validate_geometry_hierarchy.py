#!/usr/bin/env python3
"""Verify the single-device FP64 block AMG hierarchy against independent CPU sums.

Run against this fork's shared library, for example:
  python3 examples/validate_geometry_hierarchy.py --amgx ./build/libamgxsh.so
Use --cudart if libcudart is outside the loader path. No Python CUDA stack is needed.
"""

import argparse
import ctypes as ct
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass


I, F, P, Size = ct.c_int, ct.c_double, ct.c_void_p, ct.c_size_t
DDI = 8193  # AMGX_mode_dDDI in amgx_config.h
SELECTORS = (2, 4, 8)


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def array(kind, values):
    return (kind * len(values))(*values)


def function(library, name, args):
    native = getattr(library, name)
    native.argtypes = args
    native.restype = I

    def checked(*arguments):
        code = native(*arguments)
        if code != 0:
            raise RuntimeError(f"{name} failed with code {code}")
    return checked


class API:
    """Minimal checked ABI; every resource has one owner in a surrounding ExitStack."""

    def __init__(self, amgx, cudart):
        amg = ct.CDLL(amgx)
        cuda = ct.CDLL(cudart)
        amg_functions = {
            "initialize": [], "finalize": [], "initialize_plugins": [], "finalize_plugins": [],
            "config_create": [ct.POINTER(P), ct.c_char_p], "config_destroy": [P],
            "resources_create": [ct.POINTER(P), P, P, I, ct.POINTER(I)], "resources_destroy": [P],
            "matrix_create": [ct.POINTER(P), P, I], "matrix_destroy": [P],
            "matrix_upload_all": [P, I, I, I, I, P, P, P, P],
            "matrix_attach_edge_weights": [P, I, P], "matrix_replace_coefficients": [P, I, I, P, P],
            "solver_create": [ct.POINTER(P), P, I, P], "solver_destroy": [P],
            "solver_setup": [P, P], "solver_resetup": [P, P],
            "solver_get_hierarchy_memory": [P, I, P, ct.POINTER(I)],
            "solver_download_level_field": [P, I, I, Size, P, ct.POINTER(Size)],
            "vector_create": [ct.POINTER(P), P, I], "vector_destroy": [P],
            "vector_bind": [P, P], "vector_upload": [P, I, I, P], "vector_download": [P, P],
            "vector_set_zero": [P, I, I], "solver_solve_with_0_initial_guess": [P, P, P],
        }
        for name, args in amg_functions.items():
            setattr(self, name, function(amg, "AMGX_" + name, args))
        self.set_device = function(cuda, "cudaSetDevice", [I])
        self.malloc = function(cuda, "cudaMalloc", [ct.POINTER(P), Size])
        self.copy = function(cuda, "cudaMemcpy", [P, P, Size, I])
        self.free = function(cuda, "cudaFree", [P])

    def own(self, stack, kind, *args):
        obj = P()
        getattr(self, kind + "_create")(ct.byref(obj), *args)
        stack.callback(getattr(self, kind + "_destroy"), obj)
        return obj

    def field(self, solver, level, number, scalar):
        size = Size()
        read = self.solver_download_level_field
        read(solver, level, number, 0, None, ct.byref(size))
        require(size.value % ct.sizeof(scalar) == 0, "misaligned diagnostic field")
        data = (scalar * (size.value // ct.sizeof(scalar)))()
        if size.value:
            capacity = size.value
            read(solver, level, number, capacity, data, ct.byref(size))
            require(size.value == capacity, "diagnostic field changed during download")
        return tuple(data)


@dataclass
class Graph:
    offsets: tuple
    columns: tuple
    blocks: dict  # (row, column) -> four row-major entries, including diagonal
    weights: dict  # only off-diagonal connections
    aggregates: tuple = ()

    @property
    def rows(self):
        return len(self.offsets) - 1

    def packed(self):
        off, diag = [], []
        for row in range(self.rows):
            diag.extend(self.blocks[row, row])
            for slot in range(self.offsets[row], self.offsets[row + 1]):
                off.extend(self.blocks[row, self.columns[slot]])
        return array(F, off), array(F, diag)


def torus(width, update):
    """Four-neighbor torus. Geometric strengths are symmetric but differ per edge;
    block transport is directed, with nonzero unequal component couplings."""
    rows = width * width
    blocks, weights, offsets, columns = {}, {}, [0], []
    for i in range(rows):
        x, y = i % width, i // width
        adjacent = [width*y + (x+1) % width, width*((y+1) % width) + x,
                    width*y + (x-1) % width, width*((y-1) % width) + x]
        assert len(set(adjacent)) == 4
        for direction, j in enumerate(adjacent):
            a, b = sorted((i, j))
            w = 1.0 + (a*rows + b) / (rows*rows) + (0.6 if direction % 2 == 0 else 0.)
            weights[i, j] = w
            # Distinct changes to off-diagonal and diagonal entries expose stale RAP/smoother data.
            k = (1. + .23*update) * (1. + .025 * (i % 7) + .07*direction)
            blocks[i, j] = (-.56*k, .039*k, -.062*k, -.43*k)
        blocks[i, i] = (3.9 + .009 * (i % 17) + .6*update, -.12 + .02*update,
                        .21 - .03*update, 3.2 + .013*(i % 11) + .31*update)
        columns.extend(adjacent)
        offsets.append(len(columns))
    return Graph(tuple(offsets), tuple(columns), blocks, weights)


def check_values(actual, expected, label):
    require(len(actual) == len(expected), f"{label}: length mismatch")
    for j, (a, b) in enumerate(zip(actual, expected)):
        require(math.isclose(a, b, rel_tol=3e-11, abs_tol=3e-12),
                f"{label}[{j}]: got {a:.17g}, expected {b:.17g}")


def check_graph(actual, expected, label):
    for attribute in ("blocks", "weights"):
        got, want = getattr(actual, attribute), getattr(expected, attribute)
        require(got.keys() == want.keys(), f"{label}: {attribute} adjacency differs")
        for edge, value in want.items():
            if attribute == "weights":
                check_values((got[edge],), (value,), f"{label} strength {edge}")
            else:
                check_values(got[edge], value, f"{label} block {edge}")


def get_levels(api, solver):
    count = I()
    api.solver_get_hierarchy_memory(solver, 0, None, ct.byref(count))
    require(count.value >= 2, "grid did not generate a coarse level")
    levels = []
    for level in range(count.value):
        offsets, columns, aggregates, values, strength = [
            api.field(solver, level, n, t) for n, t in enumerate((I, I, I, F, F))]
        rows, nnz = len(offsets)-1, len(columns)
        require(offsets and offsets[0] == 0 and offsets[-1] == nnz and
                all(a <= b for a, b in zip(offsets, offsets[1:])), f"invalid CSR, level {level}")
        separate = len(values) == 4*(nnz+rows)
        require(separate or len(values) == 4*nnz, "unsupported diagnostic block layout")
        require(len(strength) == nnz, f"missing strengths, level {level}")
        require(len(aggregates) == (rows if level+1 < count.value else 0), "invalid aggregate extent")
        blocks, weights = {}, {}
        for i in range(rows):
            for slot in range(offsets[i], offsets[i+1]):
                j = columns[slot]
                require(0 <= j < rows and (i, j) not in blocks, "invalid or duplicate CSR column")
                blocks[i, j] = values[4*slot:4*(slot+1)]
                if i != j:
                    require(strength[slot] > 0 and math.isfinite(strength[slot]), "invalid geometric strength")
                    weights[i, j] = strength[slot]
                else:
                    require(strength[slot] == 0, "inline diagonal entered matching")
            if separate:
                require((i, i) not in blocks, "duplicate separate diagonal")
                blocks[i, i] = values[4*(nnz+i):4*(nnz+i+1)]
        levels.append(Graph(offsets, columns, blocks, weights, aggregates))
    return levels


def check_hierarchy(levels, finest):
    """Independent scalar coarsening, keyed only on the downloaded aggregation map."""
    reference = finest
    multi_edges = 0
    for level, actual in enumerate(levels):
        check_graph(actual, reference, f"level {level}")
        if not actual.aggregates:
            break
        agg = actual.aggregates
        coarse = levels[level+1]
        require(set(agg) == set(range(coarse.rows)), f"invalid aggregate IDs, level {level}")
        blocks, edges = {}, {}
        for (i, j), block in reference.blocks.items():
            pair = agg[i], agg[j]
            dest = blocks.setdefault(pair, [0.] * 4)
            for k in range(4):
                dest[k] += block[k]
        for (i, j), w in reference.weights.items():
            if agg[i] != agg[j]:
                edges.setdefault((agg[i], agg[j]), []).append(w)
        multi_edges += sum(len(contributions) > 1 for contributions in edges.values())
        reference = Graph(coarse.offsets, coarse.columns, blocks,
                          {pair: math.fsum(w) for pair, w in edges.items()})
    require(multi_edges > 0, "fixture did not exercise multiple crossing edges")


def cycle(api, stack, solver, matrix, resources, rhs, graph):
    right = api.own(stack, "vector", resources, DDI)
    sol = api.own(stack, "vector", resources, DDI)
    api.vector_bind(right, matrix)
    api.vector_bind(sol, matrix)
    api.vector_upload(right, graph.rows, 2, array(F, rhs))
    api.vector_set_zero(sol, graph.rows, 2)
    api.solver_solve_with_0_initial_guess(solver, right, sol)
    x = (F * len(rhs))()
    api.vector_download(sol, x)
    residual = list(rhs)
    for (i, j), (a, b, c, d) in graph.blocks.items():
        residual[2*i] -= a*x[2*j] + b*x[2*j+1]
        residual[2*i+1] -= c*x[2*j] + d*x[2*j+1]
    ratio = math.sqrt(math.fsum(v*v for v in residual) / math.fsum(v*v for v in rhs))
    require(ratio < 1, f"cycle did not reduce true residual (ratio {ratio})")
    return tuple(x), ratio


def verify(api, device, width, selector):
    config = {"config_version": 2, "determinism_flag": 1, "block_format": "ROW_MAJOR",
              "solver": {"solver": "AMG", "algorithm": "AGGREGATION", "selector": f"SIZE_{selector}",
                         "weight_formula": 2, "structure_reuse_levels": -1, "cycle": "V",
                         "smoother": {"solver": "BLOCK_JACOBI", "relaxation_factor": .75,
                                      "monitor_residual": 0, "print_solve_stats": 0},
                         "coarse_solver": "NOSOLVER", "monitor_residual": 0, "print_solve_stats": 0,
                         "print_grid_stats": 0, "max_iters": 1, "presweeps": 2, "postsweeps": 2,
                         "coarsest_sweeps": 8, "min_coarse_rows": 2, "max_levels": 10}}
    initial, updated = torus(width, 0), torus(width, 1)
    rhs = [math.sin(.23*i) + .5*math.cos(.07*i) for i in range(2*initial.rows)]
    with ExitStack() as stack:
        cfg = api.own(stack, "config", json.dumps(config).encode())
        resources = api.own(stack, "resources", cfg, None, 1, ct.byref(I(device)))
        device_weights = P()
        flat = array(F, [initial.weights[i, j] for i in range(initial.rows)
                         for j in initial.columns[initial.offsets[i]:initial.offsets[i+1]]])
        api.malloc(ct.byref(device_weights), ct.sizeof(flat))
        stack.callback(api.free, device_weights)
        api.copy(device_weights, flat, ct.sizeof(flat), 1)  # host to device

        def setup(graph):
            matrix = api.own(stack, "matrix", resources, DDI)
            values, diagonal = graph.packed()
            api.matrix_upload_all(matrix, graph.rows, len(graph.columns), 2, 2,
                                  array(I, graph.offsets), array(I, graph.columns), values, diagonal)
            api.matrix_attach_edge_weights(matrix, len(flat), device_weights)
            solver = api.own(stack, "solver", resources, DDI, cfg)
            api.solver_setup(solver, matrix)
            return solver, matrix

        solver, matrix = setup(initial)
        before = get_levels(api, solver)
        check_hierarchy(before, initial)
        _, initial_ratio = cycle(api, stack, solver, matrix, resources, rhs, initial)
        data, diag = updated.packed()
        api.matrix_replace_coefficients(matrix, updated.rows, len(updated.columns), data, diag)
        api.solver_resetup(solver, matrix)
        after = get_levels(api, solver)
        check_hierarchy(after, updated)
        require(len(before) == len(after), "resetup changed level count")
        for level, (old, new) in enumerate(zip(before, after)):
            require((old.offsets, old.columns, old.aggregates, old.weights) ==
                    (new.offsets, new.columns, new.aggregates, new.weights),
                    f"resetup changed CSR order, aggregate membership or geometric strength at level {level}")
        x, ratio = cycle(api, stack, solver, matrix, resources, rhs, updated)
        fresh, fresh_matrix = setup(updated)
        ref_levels = get_levels(api, fresh)
        check_hierarchy(ref_levels, updated)
        ref_x, ref_ratio = cycle(api, stack, fresh, fresh_matrix, resources, rhs, updated)
        check_values(x, ref_x, "refreshed and independently rebuilt cycle")
        check_values((ratio,), (ref_ratio,), "true residual reduction")
        print(json.dumps({"selector": f"SIZE_{selector}", "rows": [lev.rows for lev in after],
                          "initial_relative_residual": initial_ratio,
                          "refreshed_relative_residual": ratio, "rebuilt_relative_residual": ref_ratio,
                          "max_cycle_difference": max(abs(a-b) for a, b in zip(x, ref_x))}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amgx", default="libamgxsh.so")
    parser.add_argument("--cudart", default="libcudart.so")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--selector", nargs="+", type=int, choices=SELECTORS, default=SELECTORS)
    args = parser.parse_args()
    require(args.width >= 8, "width must be at least 8 for multilevel matching")
    api = API(args.amgx, args.cudart)
    api.set_device(args.device)
    with ExitStack() as stack:
        api.initialize()
        stack.callback(api.finalize)
        api.initialize_plugins()
        stack.callback(api.finalize_plugins)
        for size in args.selector:
            verify(api, args.device, args.width, size)


if __name__ == "__main__":
    main()

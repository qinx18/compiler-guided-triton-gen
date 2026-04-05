#!/usr/bin/env python3
"""Measure total application speedup including CPU<->GPU data transfer.

For each kernel, measures:
  - C reference: kernel time (data already in CPU memory)
  - Triton kernel-only: GPU kernel time (data already on GPU)
  - Triton total: CPU->GPU transfer + kernel + GPU->CPU transfer

This gives the "total application speedup" for scenarios where data
starts and ends on CPU.
"""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path
import torch
import json

sys.path.append(str(Path(__file__).parent))

# ==========================================================================
# SpMV (miniFE)
# ==========================================================================
def measure_spmv():
    from realworld_results.llm_triton.spmv.attempt3 import spmv_triton

    NROWS = 4096
    NNZ_PER_ROW = 27
    NNZ = NROWS * NNZ_PER_ROW
    NCOLS = NROWS

    # Build CSR structure on CPU
    row_offsets_np = np.arange(0, (NROWS + 1) * NNZ_PER_ROW, NNZ_PER_ROW, dtype=np.int32)
    cols_list = []
    half_bw = NNZ_PER_ROW // 2
    for r in range(NROWS):
        col_start = max(0, r - half_bw)
        row_cols = list(range(col_start, min(col_start + NNZ_PER_ROW, NCOLS)))
        while len(row_cols) < NNZ_PER_ROW:
            row_cols.append(row_cols[-1])
        cols_list.extend(row_cols[:NNZ_PER_ROW])
    cols_np = np.array(cols_list, dtype=np.int32)
    vals_np = np.random.randn(NNZ).astype(np.float32) * 0.1
    x_np = np.random.randn(NCOLS).astype(np.float32)
    y_np = np.zeros(NROWS, dtype=np.float32)

    # C reference benchmark
    lib_path = Path("c_reference/realworld_libs/libspmv.so")
    lib = ctypes.CDLL(str(lib_path))

    num_warmup = 5
    num_iter = 50

    # Warmup
    for _ in range(num_warmup):
        CType_ro = (ctypes.c_int * (NROWS + 1))
        c_ro = CType_ro.in_dll(lib, 'row_offsets')
        ctypes.memmove(c_ro, row_offsets_np.ctypes.data, row_offsets_np.nbytes)
        CType_cols = (ctypes.c_int * NNZ)
        c_cols = CType_cols.in_dll(lib, 'cols')
        ctypes.memmove(c_cols, cols_np.ctypes.data, cols_np.nbytes)
        CType_vals = (ctypes.c_float * NNZ)
        c_vals = CType_vals.in_dll(lib, 'vals')
        ctypes.memmove(c_vals, vals_np.ctypes.data, vals_np.nbytes)
        CType_x = (ctypes.c_float * NCOLS)
        c_x = CType_x.in_dll(lib, 'x')
        ctypes.memmove(c_x, x_np.ctypes.data, x_np.nbytes)
        lib.spmv_kernel()

    start = time.perf_counter()
    for _ in range(num_iter):
        ctypes.memmove(c_ro, row_offsets_np.ctypes.data, row_offsets_np.nbytes)
        ctypes.memmove(c_cols, cols_np.ctypes.data, cols_np.nbytes)
        ctypes.memmove(c_vals, vals_np.ctypes.data, vals_np.nbytes)
        ctypes.memmove(c_x, x_np.ctypes.data, x_np.nbytes)
        lib.spmv_kernel()
    c_time = (time.perf_counter() - start) / num_iter

    # Triton kernel-only benchmark (data already on GPU)
    row_offsets_gpu = torch.from_numpy(row_offsets_np).cuda()
    cols_gpu = torch.from_numpy(cols_np).cuda()
    vals_gpu = torch.from_numpy(vals_np).cuda()
    x_gpu = torch.from_numpy(x_np).cuda()
    y_gpu = torch.zeros(NROWS, device='cuda', dtype=torch.float32)

    for _ in range(num_warmup):
        y_tr = y_gpu.clone()
        spmv_triton(cols_gpu, row_offsets_gpu, vals_gpu, x_gpu, y_tr, NNZ_PER_ROW, NROWS)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        y_tr = y_gpu.clone()
        spmv_triton(cols_gpu, row_offsets_gpu, vals_gpu, x_gpu, y_tr, NNZ_PER_ROW, NROWS)
    torch.cuda.synchronize()
    kernel_time = (time.perf_counter() - start) / num_iter

    # Triton total benchmark (CPU -> GPU -> kernel -> GPU -> CPU)
    for _ in range(num_warmup):
        ro_g = torch.from_numpy(row_offsets_np).cuda()
        co_g = torch.from_numpy(cols_np).cuda()
        va_g = torch.from_numpy(vals_np).cuda()
        x_g = torch.from_numpy(x_np).cuda()
        y_g = torch.zeros(NROWS, device='cuda', dtype=torch.float32)
        spmv_triton(co_g, ro_g, va_g, x_g, y_g, NNZ_PER_ROW, NROWS)
        result = y_g.cpu().numpy()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        ro_g = torch.from_numpy(row_offsets_np).cuda()
        co_g = torch.from_numpy(cols_np).cuda()
        va_g = torch.from_numpy(vals_np).cuda()
        x_g = torch.from_numpy(x_np).cuda()
        y_g = torch.zeros(NROWS, device='cuda', dtype=torch.float32)
        spmv_triton(co_g, ro_g, va_g, x_g, y_g, NNZ_PER_ROW, NROWS)
        result = y_g.cpu().numpy()
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) / num_iter

    return {
        'kernel': 'spmv',
        'c_time_ms': c_time * 1000,
        'triton_kernel_ms': kernel_time * 1000,
        'triton_total_ms': total_time * 1000,
        'kernel_speedup': c_time / kernel_time if kernel_time > 0 else 0,
        'total_speedup': c_time / total_time if total_time > 0 else 0,
    }


# ==========================================================================
# LJ Force (miniMD)
# ==========================================================================
def measure_lj_force():
    from realworld_results.llm_triton.lj_force.attempt1 import lj_force_triton

    NLOCAL = 4000
    MAXNEIGHS = 128
    PAD = 3

    # Build positions on lattice
    lattice_n = int(round(NLOCAL ** (1.0/3.0))) + 1
    spacing = 2.0
    coords = []
    for ix in range(lattice_n):
        for iy in range(lattice_n):
            for iz in range(lattice_n):
                if len(coords) >= NLOCAL:
                    break
                coords.append([ix*spacing + np.random.uniform(-0.2, 0.2),
                               iy*spacing + np.random.uniform(-0.2, 0.2),
                               iz*spacing + np.random.uniform(-0.2, 0.2)])
            if len(coords) >= NLOCAL:
                break
        if len(coords) >= NLOCAL:
            break
    coords = np.array(coords[:NLOCAL], dtype=np.float32)
    pos_np = coords.flatten()
    f_np = np.zeros(NLOCAL * PAD, dtype=np.float32)

    # Build neighbor lists
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    cutoff = 4.0
    numneigh_list = []
    neighbors_flat = []
    for i_atom in range(NLOCAL):
        nlist = tree.query_ball_point(coords[i_atom], cutoff)
        nlist = [j for j in nlist if j != i_atom]
        if len(nlist) > MAXNEIGHS:
            nlist = nlist[:MAXNEIGHS]
        numneigh_list.append(len(nlist))
        padded = nlist + [0] * (MAXNEIGHS - len(nlist))
        neighbors_flat.extend(padded)
    numneigh_np = np.array(numneigh_list, dtype=np.int32)
    neighbors_np = np.array(neighbors_flat, dtype=np.int32)

    cutforcesq_val = 16.0
    sigma6_val = 1.0
    epsilon_val = 1.0

    # C reference
    lib_path = Path("c_reference/realworld_libs/liblj_force.so")
    lib = ctypes.CDLL(str(lib_path))

    num_warmup = 5
    num_iter = 50

    CType_pos = (ctypes.c_float * (NLOCAL * PAD))
    c_pos = CType_pos.in_dll(lib, 'pos')
    CType_f = (ctypes.c_float * (NLOCAL * PAD))
    c_f = CType_f.in_dll(lib, 'f')
    CType_nei = (ctypes.c_int * (NLOCAL * MAXNEIGHS))
    c_nei = CType_nei.in_dll(lib, 'neighbors')
    CType_nn = (ctypes.c_int * NLOCAL)
    c_nn = CType_nn.in_dll(lib, 'numneigh')

    for _ in range(num_warmup):
        ctypes.memmove(c_pos, pos_np.ctypes.data, pos_np.nbytes)
        ctypes.memmove(c_nei, neighbors_np.ctypes.data, neighbors_np.nbytes)
        ctypes.memmove(c_nn, numneigh_np.ctypes.data, numneigh_np.nbytes)
        ctypes.c_float.in_dll(lib, 'cutforcesq_val').value = cutforcesq_val
        ctypes.c_float.in_dll(lib, 'sigma6_val').value = sigma6_val
        ctypes.c_float.in_dll(lib, 'epsilon_val').value = epsilon_val
        lib.lj_force_kernel()

    start = time.perf_counter()
    for _ in range(num_iter):
        ctypes.memmove(c_pos, pos_np.ctypes.data, pos_np.nbytes)
        ctypes.memmove(c_nei, neighbors_np.ctypes.data, neighbors_np.nbytes)
        ctypes.memmove(c_nn, numneigh_np.ctypes.data, numneigh_np.nbytes)
        lib.lj_force_kernel()
    c_time = (time.perf_counter() - start) / num_iter

    # Triton kernel-only
    pos_gpu = torch.from_numpy(pos_np).cuda()
    f_gpu = torch.zeros(NLOCAL * PAD, device='cuda', dtype=torch.float32)
    neighbors_gpu = torch.from_numpy(neighbors_np).cuda()
    numneigh_gpu = torch.from_numpy(numneigh_np).cuda()

    for _ in range(num_warmup):
        f_tr = f_gpu.clone()
        lj_force_triton(f_tr, neighbors_gpu, numneigh_gpu, pos_gpu, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        f_tr = f_gpu.clone()
        lj_force_triton(f_tr, neighbors_gpu, numneigh_gpu, pos_gpu, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD)
    torch.cuda.synchronize()
    kernel_time = (time.perf_counter() - start) / num_iter

    # Triton total (CPU -> GPU -> kernel -> GPU -> CPU)
    for _ in range(num_warmup):
        pos_g = torch.from_numpy(pos_np).cuda()
        f_g = torch.zeros(NLOCAL * PAD, device='cuda', dtype=torch.float32)
        nei_g = torch.from_numpy(neighbors_np).cuda()
        nn_g = torch.from_numpy(numneigh_np).cuda()
        lj_force_triton(f_g, nei_g, nn_g, pos_g, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD)
        result = f_g.cpu().numpy()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        pos_g = torch.from_numpy(pos_np).cuda()
        f_g = torch.zeros(NLOCAL * PAD, device='cuda', dtype=torch.float32)
        nei_g = torch.from_numpy(neighbors_np).cuda()
        nn_g = torch.from_numpy(numneigh_np).cuda()
        lj_force_triton(f_g, nei_g, nn_g, pos_g, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD)
        result = f_g.cpu().numpy()
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) / num_iter

    return {
        'kernel': 'lj_force',
        'c_time_ms': c_time * 1000,
        'triton_kernel_ms': kernel_time * 1000,
        'triton_total_ms': total_time * 1000,
        'kernel_speedup': c_time / kernel_time if kernel_time > 0 else 0,
        'total_speedup': c_time / total_time if total_time > 0 else 0,
    }


# ==========================================================================
# SRAD (Rodinia) — high kernel speedup, check if transfer matters
# ==========================================================================
def measure_srad():
    from realworld_results.llm_triton.srad.attempt1 import srad_triton

    ROWS, COLS, NITER = 512, 512, 10
    R1, R2, C1, C2 = 0, 127, 0, 127
    SIZE_I = ROWS * COLS

    # CPU data
    J_np = np.exp(np.random.randn(SIZE_I).astype(np.float32).clip(-3, 3))
    iN_np = np.arange(ROWS, dtype=np.int32); iN_np[0] = 0; iN_np[1:] = np.arange(ROWS-1, dtype=np.int32)
    iS_np = np.arange(ROWS, dtype=np.int32); iS_np[:-1] = np.arange(1, ROWS, dtype=np.int32); iS_np[-1] = ROWS-1
    jW_np = np.arange(COLS, dtype=np.int32); jW_np[0] = 0; jW_np[1:] = np.arange(COLS-1, dtype=np.int32)
    jE_np = np.arange(COLS, dtype=np.int32); jE_np[:-1] = np.arange(1, COLS, dtype=np.int32); jE_np[-1] = COLS-1
    c_np = np.zeros(SIZE_I, dtype=np.float32)
    dN_np = np.zeros(SIZE_I, dtype=np.float32)
    dS_np = np.zeros(SIZE_I, dtype=np.float32)
    dW_np = np.zeros(SIZE_I, dtype=np.float32)
    dE_np = np.zeros(SIZE_I, dtype=np.float32)
    lambda_val = 0.5

    # C reference
    lib = ctypes.CDLL(str(Path("c_reference/realworld_libs/libsrad.so")))
    num_warmup = 5
    num_iter = 50

    def set_c_arrays():
        for name, arr in [('J', J_np), ('iN', iN_np), ('iS', iS_np), ('jW', jW_np), ('jE', jE_np),
                          ('c', c_np), ('dN', dN_np), ('dS', dS_np), ('dW', dW_np), ('dE', dE_np)]:
            ctype = ctypes.c_float if arr.dtype == np.float32 else ctypes.c_int
            CType = (ctype * len(arr))
            c_arr = CType.in_dll(lib, name)
            ctypes.memmove(c_arr, arr.ctypes.data, arr.nbytes)
        ctypes.c_float.in_dll(lib, 'lambda_val').value = lambda_val

    for _ in range(num_warmup):
        set_c_arrays()
        lib.srad_kernel()
    start = time.perf_counter()
    for _ in range(num_iter):
        set_c_arrays()
        lib.srad_kernel()
    c_time = (time.perf_counter() - start) / num_iter

    # Triton kernel-only
    J_gpu = torch.from_numpy(J_np).cuda()
    iN_gpu = torch.from_numpy(iN_np).cuda()
    iS_gpu = torch.from_numpy(iS_np).cuda()
    jW_gpu = torch.from_numpy(jW_np).cuda()
    jE_gpu = torch.from_numpy(jE_np).cuda()
    c_gpu = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
    dN_gpu = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
    dS_gpu = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
    dW_gpu = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
    dE_gpu = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)

    for _ in range(num_warmup):
        J_tr = J_gpu.clone()
        c_tr, dN_tr, dS_tr, dW_tr, dE_tr = c_gpu.clone(), dN_gpu.clone(), dS_gpu.clone(), dW_gpu.clone(), dE_gpu.clone()
        srad_triton(J_tr, c_tr, dE_tr, dN_tr, dS_tr, dW_tr, iN_gpu, iS_gpu, jE_gpu, jW_gpu, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        J_tr = J_gpu.clone()
        c_tr, dN_tr, dS_tr, dW_tr, dE_tr = c_gpu.clone(), dN_gpu.clone(), dS_gpu.clone(), dW_gpu.clone(), dE_gpu.clone()
        srad_triton(J_tr, c_tr, dE_tr, dN_tr, dS_tr, dW_tr, iN_gpu, iS_gpu, jE_gpu, jW_gpu, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS)
    torch.cuda.synchronize()
    kernel_time = (time.perf_counter() - start) / num_iter

    # Triton total (CPU -> GPU -> kernel -> GPU -> CPU)
    for _ in range(num_warmup):
        J_g = torch.from_numpy(J_np).cuda()
        iN_g = torch.from_numpy(iN_np).cuda()
        iS_g = torch.from_numpy(iS_np).cuda()
        jW_g = torch.from_numpy(jW_np).cuda()
        jE_g = torch.from_numpy(jE_np).cuda()
        c_g = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
        dN_g = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
        dS_g = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
        dW_g = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
        dE_g = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
        srad_triton(J_g, c_g, dE_g, dN_g, dS_g, dW_g, iN_g, iS_g, jE_g, jW_g, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS)
        result = J_g.cpu().numpy()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        J_g = torch.from_numpy(J_np).cuda()
        iN_g = torch.from_numpy(iN_np).cuda()
        iS_g = torch.from_numpy(iS_np).cuda()
        jW_g = torch.from_numpy(jW_np).cuda()
        jE_g = torch.from_numpy(jE_np).cuda()
        c_g = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
        dN_g = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
        dS_g = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
        dW_g = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
        dE_g = torch.zeros(SIZE_I, device='cuda', dtype=torch.float32)
        srad_triton(J_g, c_g, dE_g, dN_g, dS_g, dW_g, iN_g, iS_g, jE_g, jW_g, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS)
        result = J_g.cpu().numpy()
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) / num_iter

    return {
        'kernel': 'srad',
        'c_time_ms': c_time * 1000,
        'triton_kernel_ms': kernel_time * 1000,
        'triton_total_ms': total_time * 1000,
        'kernel_speedup': c_time / kernel_time if kernel_time > 0 else 0,
        'total_speedup': c_time / total_time if total_time > 0 else 0,
    }


if __name__ == '__main__':
    results = {}

    for name, func in [('spmv', measure_spmv), ('lj_force', measure_lj_force), ('srad', measure_srad)]:
        print(f"\n{'='*60}")
        print(f"Measuring: {name}")
        print(f"{'='*60}")
        try:
            r = func()
            results[name] = r
            print(f"  C reference:       {r['c_time_ms']:8.3f} ms")
            print(f"  Triton kernel:     {r['triton_kernel_ms']:8.3f} ms")
            print(f"  Triton total:      {r['triton_total_ms']:8.3f} ms")
            print(f"  Kernel speedup:    {r['kernel_speedup']:8.2f}x")
            print(f"  Total speedup:     {r['total_speedup']:8.2f}x")
            transfer_overhead = r['triton_total_ms'] - r['triton_kernel_ms']
            print(f"  Transfer overhead: {transfer_overhead:8.3f} ms ({transfer_overhead/r['triton_total_ms']*100:.1f}% of total)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Kernel':<12} {'C (ms)':>8} {'Kern (ms)':>10} {'Total (ms)':>11} {'Kern spdup':>11} {'Total spdup':>12}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<12} {r['c_time_ms']:8.3f} {r['triton_kernel_ms']:10.3f} {r['triton_total_ms']:11.3f} {r['kernel_speedup']:10.2f}x {r['total_speedup']:11.2f}x")

    with open('realworld_results/total_speedup_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: realworld_results/total_speedup_results.json")

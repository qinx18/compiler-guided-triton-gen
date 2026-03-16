#!/usr/bin/env python3
"""Performance Benchmark for lavaMD (Real-World)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from realworld_results.llm_triton.lavaMD.attempt1 import lavaMD_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "realworld_libs" / "liblavaMD.so"

def run_c_reference(box_nei_c, box_nn_c, box_offset_c, fv_v_c, fv_x_c, fv_y_c, fv_z_c, qv_c, rv_v_c, rv_x_c, rv_y_c, rv_z_c, alpha_val, BOXES1D, MAX_NEIGHBORS, NUMBER_PAR_PER_BOX):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_box_nei = ctypes.c_int * (1664)
    c_arr_box_nei = CType_box_nei.in_dll(lib, 'box_nei')
    src_box_nei = np.ascontiguousarray(box_nei_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_box_nei, src_box_nei.ctypes.data, src_box_nei.nbytes)
    CType_box_nn = ctypes.c_int * (64)
    c_arr_box_nn = CType_box_nn.in_dll(lib, 'box_nn')
    src_box_nn = np.ascontiguousarray(box_nn_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_box_nn, src_box_nn.ctypes.data, src_box_nn.nbytes)
    CType_box_offset = ctypes.c_int * (64)
    c_arr_box_offset = CType_box_offset.in_dll(lib, 'box_offset')
    src_box_offset = np.ascontiguousarray(box_offset_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_box_offset, src_box_offset.ctypes.data, src_box_offset.nbytes)
    CType_fv_v = ctypes.c_double * (6400)
    c_arr_fv_v = CType_fv_v.in_dll(lib, 'fv_v')
    src_fv_v = np.ascontiguousarray(fv_v_c.astype(np.float64), dtype=np.float64)
    ctypes.memmove(c_arr_fv_v, src_fv_v.ctypes.data, src_fv_v.nbytes)
    CType_fv_x = ctypes.c_double * (6400)
    c_arr_fv_x = CType_fv_x.in_dll(lib, 'fv_x')
    src_fv_x = np.ascontiguousarray(fv_x_c.astype(np.float64), dtype=np.float64)
    ctypes.memmove(c_arr_fv_x, src_fv_x.ctypes.data, src_fv_x.nbytes)
    CType_fv_y = ctypes.c_double * (6400)
    c_arr_fv_y = CType_fv_y.in_dll(lib, 'fv_y')
    src_fv_y = np.ascontiguousarray(fv_y_c.astype(np.float64), dtype=np.float64)
    ctypes.memmove(c_arr_fv_y, src_fv_y.ctypes.data, src_fv_y.nbytes)
    CType_fv_z = ctypes.c_double * (6400)
    c_arr_fv_z = CType_fv_z.in_dll(lib, 'fv_z')
    src_fv_z = np.ascontiguousarray(fv_z_c.astype(np.float64), dtype=np.float64)
    ctypes.memmove(c_arr_fv_z, src_fv_z.ctypes.data, src_fv_z.nbytes)
    CType_qv = ctypes.c_double * (6400)
    c_arr_qv = CType_qv.in_dll(lib, 'qv')
    src_qv = np.ascontiguousarray(qv_c.astype(np.float64), dtype=np.float64)
    ctypes.memmove(c_arr_qv, src_qv.ctypes.data, src_qv.nbytes)
    CType_rv_v = ctypes.c_double * (6400)
    c_arr_rv_v = CType_rv_v.in_dll(lib, 'rv_v')
    src_rv_v = np.ascontiguousarray(rv_v_c.astype(np.float64), dtype=np.float64)
    ctypes.memmove(c_arr_rv_v, src_rv_v.ctypes.data, src_rv_v.nbytes)
    CType_rv_x = ctypes.c_double * (6400)
    c_arr_rv_x = CType_rv_x.in_dll(lib, 'rv_x')
    src_rv_x = np.ascontiguousarray(rv_x_c.astype(np.float64), dtype=np.float64)
    ctypes.memmove(c_arr_rv_x, src_rv_x.ctypes.data, src_rv_x.nbytes)
    CType_rv_y = ctypes.c_double * (6400)
    c_arr_rv_y = CType_rv_y.in_dll(lib, 'rv_y')
    src_rv_y = np.ascontiguousarray(rv_y_c.astype(np.float64), dtype=np.float64)
    ctypes.memmove(c_arr_rv_y, src_rv_y.ctypes.data, src_rv_y.nbytes)
    CType_rv_z = ctypes.c_double * (6400)
    c_arr_rv_z = CType_rv_z.in_dll(lib, 'rv_z')
    src_rv_z = np.ascontiguousarray(rv_z_c.astype(np.float64), dtype=np.float64)
    ctypes.memmove(c_arr_rv_z, src_rv_z.ctypes.data, src_rv_z.nbytes)
    ctypes.c_double.in_dll(lib, 'alpha_val').value = float(alpha_val)
    func = getattr(lib, "lavaMD_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_fv_v = ctypes.c_double * (6400)
    c_arr_fv_v = CType_fv_v.in_dll(lib, 'fv_v')
    fv_v_c[:] = np.frombuffer(c_arr_fv_v, dtype=np.float64).reshape(6400).astype(np.float32).copy()
    CType_fv_x = ctypes.c_double * (6400)
    c_arr_fv_x = CType_fv_x.in_dll(lib, 'fv_x')
    fv_x_c[:] = np.frombuffer(c_arr_fv_x, dtype=np.float64).reshape(6400).astype(np.float32).copy()
    CType_fv_y = ctypes.c_double * (6400)
    c_arr_fv_y = CType_fv_y.in_dll(lib, 'fv_y')
    fv_y_c[:] = np.frombuffer(c_arr_fv_y, dtype=np.float64).reshape(6400).astype(np.float32).copy()
    CType_fv_z = ctypes.c_double * (6400)
    c_arr_fv_z = CType_fv_z.in_dll(lib, 'fv_z')
    fv_z_c[:] = np.frombuffer(c_arr_fv_z, dtype=np.float64).reshape(6400).astype(np.float32).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    # Use double precision (fp = double in original)
    NUMBER_BOXES = 4**3
    TOTAL = NUMBER_BOXES * 100
    # Particle positions
    rv_x = torch.randn(TOTAL, device='cuda', dtype=torch.float64) * 10.0
    rv_y = torch.randn(TOTAL, device='cuda', dtype=torch.float64) * 10.0
    rv_z = torch.randn(TOTAL, device='cuda', dtype=torch.float64) * 10.0
    rv_v = rv_x**2 + rv_y**2 + rv_z**2  # precomputed r^2
    qv = torch.randn(TOTAL, device='cuda', dtype=torch.float64)
    fv_v = torch.zeros(TOTAL, device='cuda', dtype=torch.float64)
    fv_x = torch.zeros(TOTAL, device='cuda', dtype=torch.float64)
    fv_y = torch.zeros(TOTAL, device='cuda', dtype=torch.float64)
    fv_z = torch.zeros(TOTAL, device='cuda', dtype=torch.float64)
    # Box metadata: particle offset per box
    box_offset = torch.arange(0, TOTAL, 100, device='cuda', dtype=torch.int32)
    # Box neighbor computation
    NUMBER_BOXES = 4**3
    box_nn = torch.zeros(NUMBER_BOXES, dtype=torch.int32, device='cuda')
    box_nei = torch.zeros(NUMBER_BOXES * 26, dtype=torch.int32, device='cuda')
    for l_idx in range(NUMBER_BOXES):
        bx = l_idx % 4
        by = (l_idx // 4) % 4
        bz = l_idx // (4 * 4)
        nn = 0
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = bx + dx, by + dy, bz + dz
                    if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                        box_nei[l_idx * 26 + nn] = nz * 4*4 + ny * 4 + nx
                        nn += 1
        box_nn[l_idx] = nn
    alpha_val = 0.5
    alpha_val = 0.5
    BOXES1D = 4
    MAX_NEIGHBORS = 26
    NUMBER_PAR_PER_BOX = 100

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            box_nei_c = box_nei.cpu().numpy().copy()
            box_nn_c = box_nn.cpu().numpy().copy()
            box_offset_c = box_offset.cpu().numpy().copy()
            fv_v_c = fv_v.cpu().numpy().copy()
            fv_x_c = fv_x.cpu().numpy().copy()
            fv_y_c = fv_y.cpu().numpy().copy()
            fv_z_c = fv_z.cpu().numpy().copy()
            qv_c = qv.cpu().numpy().copy()
            rv_v_c = rv_v.cpu().numpy().copy()
            rv_x_c = rv_x.cpu().numpy().copy()
            rv_y_c = rv_y.cpu().numpy().copy()
            rv_z_c = rv_z.cpu().numpy().copy()
            run_c_reference(box_nei_c, box_nn_c, box_offset_c, fv_v_c, fv_x_c, fv_y_c, fv_z_c, qv_c, rv_v_c, rv_x_c, rv_y_c, rv_z_c, alpha_val, BOXES1D, MAX_NEIGHBORS, NUMBER_PAR_PER_BOX)
        start = time.perf_counter()
        for _ in range(num_iterations):
            box_nei_c = box_nei.cpu().numpy().copy()
            box_nn_c = box_nn.cpu().numpy().copy()
            box_offset_c = box_offset.cpu().numpy().copy()
            fv_v_c = fv_v.cpu().numpy().copy()
            fv_x_c = fv_x.cpu().numpy().copy()
            fv_y_c = fv_y.cpu().numpy().copy()
            fv_z_c = fv_z.cpu().numpy().copy()
            qv_c = qv.cpu().numpy().copy()
            rv_v_c = rv_v.cpu().numpy().copy()
            rv_x_c = rv_x.cpu().numpy().copy()
            rv_y_c = rv_y.cpu().numpy().copy()
            rv_z_c = rv_z.cpu().numpy().copy()
            run_c_reference(box_nei_c, box_nn_c, box_offset_c, fv_v_c, fv_x_c, fv_y_c, fv_z_c, qv_c, rv_v_c, rv_x_c, rv_y_c, rv_z_c, alpha_val, BOXES1D, MAX_NEIGHBORS, NUMBER_PAR_PER_BOX)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            box_nei_tr = box_nei.clone()
            box_nn_tr = box_nn.clone()
            box_offset_tr = box_offset.clone()
            fv_v_tr = fv_v.clone()
            fv_x_tr = fv_x.clone()
            fv_y_tr = fv_y.clone()
            fv_z_tr = fv_z.clone()
            qv_tr = qv.clone()
            rv_v_tr = rv_v.clone()
            rv_x_tr = rv_x.clone()
            rv_y_tr = rv_y.clone()
            rv_z_tr = rv_z.clone()
            lavaMD_triton(box_nei_tr, box_nn_tr, box_offset_tr, fv_v_tr, fv_x_tr, fv_y_tr, fv_z_tr, qv_tr, rv_v_tr, rv_x_tr, rv_y_tr, rv_z_tr, alpha_val, BOXES1D, MAX_NEIGHBORS, NUMBER_PAR_PER_BOX)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            box_nei_tr = box_nei.clone()
            box_nn_tr = box_nn.clone()
            box_offset_tr = box_offset.clone()
            fv_v_tr = fv_v.clone()
            fv_x_tr = fv_x.clone()
            fv_y_tr = fv_y.clone()
            fv_z_tr = fv_z.clone()
            qv_tr = qv.clone()
            rv_v_tr = rv_v.clone()
            rv_x_tr = rv_x.clone()
            rv_y_tr = rv_y.clone()
            rv_z_tr = rv_z.clone()
            lavaMD_triton(box_nei_tr, box_nn_tr, box_offset_tr, fv_v_tr, fv_x_tr, fv_y_tr, fv_z_tr, qv_tr, rv_v_tr, rv_x_tr, rv_y_tr, rv_z_tr, alpha_val, BOXES1D, MAX_NEIGHBORS, NUMBER_PAR_PER_BOX)
        torch.cuda.synchronize()
        tr_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"Triton error: {e}")

    # Report
    speedup = c_time / tr_time if c_time and tr_time and tr_time > 0 else None
    c_ms = c_time * 1000 if c_time else -1
    tr_ms = tr_time * 1000 if tr_time else -1
    sp = speedup if speedup else -1

    print(f"C ref:   {c_ms:8.3f} ms")
    print(f"Triton:  {tr_ms:8.3f} ms")
    if speedup:
        print(f"Speedup: {speedup:8.2f}x")
    else:
        print(f"Speedup: N/A")
    print(f"BENCHMARK_RESULT:{c_ms:.6f},{tr_ms:.6f},{sp:.6f}")

if __name__ == "__main__":
    benchmark()
